import argparse  
import os, sys  
import shutil  
import time  
from pathlib import Path  
  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)  
  
print(sys.path)  
import cv2  
import torch  
import torch.backends.cudnn as cudnn  
from numpy import random  
import scipy.special  
import numpy as np  
import torchvision.transforms as transforms  
import PIL.Image as image  
  
from lib.config import cfg  
from lib.config import update_config  
from lib.utils.utils import create_logger, select_device, time_synchronized  
from lib.models import get_net  
from lib.dataset import LoadImages, LoadStreams  
from lib.core.general import non_max_suppression, scale_coords  
from lib.utils import plot_one_box  
from lib.core.function import AverageMeter  
from lib.core.postprocess import morphological_process, connect_lane  
from tqdm import tqdm  
  
normalize = transforms.Normalize(  
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  
    )  
  
transform=transforms.Compose([  
            transforms.ToTensor(),  
            normalize,  
        ])


def test_colors():
    """测试颜色显示是否正确"""
    # 创建测试图像
    test_img = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # 绘制不同颜色的区域
    test_img[50:100, 50:150] = [0, 0, 255]    # 红色区域 (BGR)
    test_img[50:100, 200:300] = [0, 255, 255]  # 黄色区域 (BGR)  
    test_img[120:170, 50:350] = [0, 255, 0]    # 绿色区域 (BGR)
    
    # 添加标签
    cv2.putText(test_img, 'Red (Solid)', (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(test_img, 'Yellow (Dashed)', (210, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(test_img, 'Green (Drivable)', (150, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return test_img


def analyze_lane_type(lane_mask):  
    """  
    分析车道线类型：实线或虚线  
    """  
    # 使用连通组件分析  
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(  
        lane_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)  
      
    solid_mask = np.zeros_like(lane_mask)  
    dashed_mask = np.zeros_like(lane_mask)  
    
    print(f"🔍 发现 {num_labels-1} 个车道线连通组件")
    
    # 收集所有组件的特征信息用于相对比较
    components_info = []
    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]  
        if area < 50:  
            continue
        
        component_mask = (labels == i).astype(np.uint8)  
        density = np.sum(component_mask) / (w * h + 1e-6)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        components_info.append({
            'index': i,
            'component_mask': component_mask,
            'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
            'density': density,
            'aspect_ratio': aspect_ratio
        })
    
    if not components_info:
        print("⚠️ 没有找到有效的车道线组件")
        return solid_mask, dashed_mask
    
    # 按密度排序，通常虚线密度更低
    components_info.sort(key=lambda x: x['density'])
    
    # 智能分类策略：
    # 1. 如果只有1-2个组件，且密度都很低，考虑将最低密度的设为虚线
    # 2. 如果有3个或更多组件，将密度最低的1/3设为虚线
    # 3. 同时考虑位置因素（中间车道线更可能是虚线）
    
    total_components = len(components_info)
    
    for idx, comp in enumerate(components_info):
        i = comp['index']
        component_mask = comp['component_mask']
        x, y, w, h, area = comp['x'], comp['y'], comp['w'], comp['h'], comp['area']
        density = comp['density']
        aspect_ratio = comp['aspect_ratio']
        
        # 计算间断性
        discontinuity_score = calculate_discontinuity(component_mask, w, h)
        
        # 位置评分：越靠近图像中心，越可能是中央分隔虚线
        image_center_x = lane_mask.shape[1] // 2
        center_distance = abs(x + w//2 - image_center_x) / image_center_x
        position_score = 1.0 - center_distance  # 越靠近中心分数越高
        
        # 综合判断是否为虚线
        is_dashed = False
        
        # 策略1: 密度极低的肯定是虚线
        if density < 0.06:
            is_dashed = True
            reason = "密度极低"
        # 策略2: 密度较低且靠近中心
        elif density < 0.08 and position_score > 0.3:
            is_dashed = True  
            reason = "密度低+中心位置"
        # 策略3: 密度较低且间断性高
        elif density < 0.10 and discontinuity_score > 0.2:
            is_dashed = True
            reason = "密度低+高间断性"
        # 策略4: 在多组件情况下，选择密度最低的一些作为虚线
        elif total_components >= 3 and idx < total_components // 3:
            is_dashed = True
            reason = "相对最低密度"
        # 策略5: 如果只有1-2个组件，密度最低的设为虚线
        elif total_components <= 2 and idx == 0 and density < 0.12:
            is_dashed = True
            reason = "单独低密度组件"
        else:
            reason = "密度正常"
        
        if is_dashed:  
            dashed_mask[component_mask == 1] = 1
            print(f"  虚线组件 {i}: 面积={area}, 位置=({x},{y}), 尺寸=({w}x{h}), 密度={density:.3f}, 原因={reason}")
        else:  
            solid_mask[component_mask == 1] = 1  
            print(f"  实线组件 {i}: 面积={area}, 位置=({x},{y}), 尺寸=({w}x{h}), 密度={density:.3f}")
      
    print(f"✅ 实线像素总数: {np.sum(solid_mask)}")
    print(f"✅ 虚线像素总数: {np.sum(dashed_mask)}")
    
    return solid_mask, dashed_mask  
  
def is_dashed_line(component_mask, x, y, w, h):  
    """  
    判断是否为虚线  
    基于线段的长宽比、密度、间断性等特征  
    """  
    # 计算长宽比  
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)  
      
    # 计算密度（像素数量/边界框面积）  
    density = np.sum(component_mask) / (w * h + 1e-6)  
    
    # 计算组件的形状特征
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contour = contours[0]
        # 计算轮廓的凸包面积比
        hull = cv2.convexHull(contour)
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / (hull_area + 1e-6)
        
        # 计算线段的间断性 - 检查垂直方向上的连续性
        discontinuity_score = calculate_discontinuity(component_mask, w, h)
        
        # 新的虚线判断条件（放宽阈值）：
        # 1. 低密度 + 长宽比适中
        # 2. 或者中等密度 + 高间断性
        # 3. 或者低实体度（形状不规则）
        
        is_dashed = (density < 0.08) or \
                   (density < 0.12 and discontinuity_score > 0.3) or \
                   (density < 0.15 and solidity < 0.7) or \
                   (aspect_ratio > 4 and density < 0.1)
        
        print(f"    分析: 长宽比={aspect_ratio:.2f}, 密度={density:.3f}, 实体度={solidity:.3f}, 间断性={discontinuity_score:.3f} -> {'虚线' if is_dashed else '实线'}")
        
        return is_dashed
    
    # 默认情况：如果无法分析轮廓，使用简单的密度判断
    return density < 0.1

def calculate_discontinuity(component_mask, w, h):
    """
    计算组件的间断性分数
    通过分析线段在垂直方向上的连续性
    """
    if h < 10:  # 太小的组件跳过
        return 0
    
    # 将组件分成若干水平带，检查每带的像素密度
    num_bands = min(10, h // 3)
    if num_bands < 3:
        return 0
    
    band_height = h // num_bands
    band_densities = []
    
    for i in range(num_bands):
        start_y = i * band_height
        end_y = min((i + 1) * band_height, h)
        band = component_mask[start_y:end_y, :]
        band_density = np.sum(band) / (band.shape[0] * band.shape[1] + 1e-6)
        band_densities.append(band_density)
    
    # 计算密度变化的标准差，高标准差表示间断性强
    if len(band_densities) > 1:
        density_std = np.std(band_densities)
        density_mean = np.mean(band_densities)
        discontinuity = density_std / (density_mean + 1e-6)
        return min(float(discontinuity), 1.0)  # 限制在[0,1]范围内
    
    return 0  
  
def show_seg_result_with_line_type(img, masks, lane_area_mask, confidence_mask, is_demo=True):  
    """  
    显示分割结果，区分实线和虚线  
    """  
    da_seg_mask, solid_mask, dashed_mask = masks  
    
    # 创建输出图像的副本
    img_result = img.copy()
    
    # 1. 首先显示可行驶区域 - 绿色半透明
    if da_seg_mask is not None and np.any(da_seg_mask):
        green_overlay = img_result.copy()
        green_overlay[da_seg_mask == 1] = [0, 255, 0]  # BGR格式的绿色
        img_result = cv2.addWeighted(img_result, 0.7, green_overlay, 0.3, 0)
    
    # 2. 显示实线 - 红色（BGR格式）
    if solid_mask is not None and np.any(solid_mask):
        img_result[solid_mask > 0] = [0, 0, 255]  # BGR: 红色
        print(f"🔴 检测到实线像素数: {np.sum(solid_mask > 0)}")
    
    # 3. 显示虚线 - 黄色（BGR格式）  
    if dashed_mask is not None and np.any(dashed_mask):
        img_result[dashed_mask > 0] = [0, 255, 255]  # BGR: 黄色
        print(f"🟡 检测到虚线像素数: {np.sum(dashed_mask > 0)}")
    
    return img_result  
  
def detect(cfg, opt):
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger, opt.device)
    
    # 检查权重文件是否存在
    if not os.path.exists(opt.weights):
        print(f"❌ 错误: 权重文件不存在: {opt.weights}")
        print("请检查以下事项:")
        print("1. 权重文件路径是否正确")
        print("2. 权重文件是否已下载")
        print("3. 尝试使用 --weights 参数指定正确的权重文件路径")
        print("\n示例:")
        print("  python aa.py --source inference/images --weights /path/to/your/model.pth")
        return
    
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    print(f"🔄 加载权重文件: {opt.weights}")
    try:
        checkpoint = torch.load(opt.weights, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ 权重文件加载成功")
    except Exception as e:
        print(f"❌ 权重文件加载失败: {str(e)}")
        return
        
    model = model.to(device)
    if half:
        model.half()  # to FP16    # Set Dataloader
    # 检查输入源是否存在
    if not opt.source.isnumeric() and not os.path.exists(opt.source):
        print(f"❌ 错误: 输入源不存在: {opt.source}")
        print("请检查以下事项:")
        print("1. 输入路径是否正确")
        print("2. 对于图片: 确保文件夹存在且包含图片文件")
        print("3. 对于视频: 确保视频文件存在")
        print("4. 对于摄像头: 使用数字 (如: --source 0)")
        return
        
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        print(f"🎥 使用摄像头: {opt.source}")
        # batch_size = len(dataset) (不需要存储变量)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        print(f"📁 处理输入: {opt.source}")
        # batch_size = 1 (不需要存储变量)
  
    # Get names and colors  
    names = model.module.names if hasattr(model, 'module') else model.names  
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  
  
    # Run inference  
    t0 = time.time()  
  
    vid_path, vid_writer = None, None  
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img  
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once  
    model.eval()  
  
    inf_time = AverageMeter()  
    nms_time = AverageMeter()  
      
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):  
        img = transform(img).to(device)  
        img = img.half() if half else img.float()  # uint8 to fp16/32  
        if img.ndimension() == 3:  
            img = img.unsqueeze(0)  
          
        # Inference  
        t1 = time_synchronized()  
        det_out, da_seg_out, ll_seg_out = model(img)  
        t2 = time_synchronized()  
          
        inf_out, _ = det_out  
        inf_time.update(t2-t1, img.size(0))  
  
        # Apply NMS  
        t3 = time_synchronized()  
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)  
        t4 = time_synchronized()  
  
        nms_time.update(t4-t3, img.size(0))  
        det = det_pred[0]  
  
        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")  
  
        _, _, height, width = img.shape  
        h, w, _ = img_det.shape  
        pad_w, pad_h = shapes[1][1]  
        pad_w = int(pad_w)  
        pad_h = int(pad_h)  
        ratio = shapes[1][0][1]  
  
        # 处理可行驶区域分割  
        da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]  
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')  
        _, da_seg_mask = torch.max(da_seg_mask, 1)  
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()  
  
        # 处理车道线分割  
        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]  
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')  
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)  
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()  
          
        # 启用形态学处理来改善线型检测效果  
        ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)  
        ll_seg_mask = connect_lane(ll_seg_mask)  
  
        print(f"📏 处理第 {i+1} 帧，车道线像素总数: {np.sum(ll_seg_mask)}")
        
        # 新增：分析线型  
        solid_mask, dashed_mask = analyze_lane_type(ll_seg_mask)  
  
        # 使用新的可视化函数  
        img_det = show_seg_result_with_line_type(img_det, (da_seg_mask, solid_mask, dashed_mask), None, None, is_demo=True)  
  
        # 绘制检测框  
        if len(det):  
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()  
            for *xyxy, conf, cls in reversed(det):  
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'  
                plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=2)  
          
        # 添加图例 - 使用白色文字，黑色描边以提高可见性
        cv2.putText(img_det, 'Red: Solid Line', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 红色文字
        cv2.putText(img_det, 'Red: Solid Line', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # 白色描边
        
        cv2.putText(img_det, 'Yellow: Dashed Line', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # 黄色文字  
        cv2.putText(img_det, 'Yellow: Dashed Line', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # 白色描边
        
        cv2.putText(img_det, 'Green: Drivable Area', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 绿色文字
        cv2.putText(img_det, 'Green: Drivable Area', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # 白色描边  
          
        if dataset.mode == 'images':  
            cv2.imwrite(save_path, img_det)  
  
        elif dataset.mode == 'video':  
            if vid_path != save_path:  # new video  
                vid_path = save_path  
                if isinstance(vid_writer, cv2.VideoWriter):  
                    vid_writer.release()  # release previous video writer  
  
                fps = vid_cap.get(cv2.CAP_PROP_FPS)  
                h, w, _ = img_det.shape  
                # 修复 VideoWriter 兼容性问题 - 使用直接的fourcc编码
                fourcc_code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
                vid_writer = cv2.VideoWriter(save_path, fourcc_code, fps, (w, h))  
            if vid_writer is not None:
                vid_writer.write(img_det)  
          
        else:  
            cv2.imshow('YOLOP - Lane Detection', img_det)  
            if cv2.waitKey(1) == ord('q'):  # 按q键退出  
                break  
  
    print('Results saved to %s' % Path(opt.save_dir))  
    print('Done. (%.3fs)' % (time.time() - t0))  
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))  
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOP车道线检测 - 支持实线虚线区分显示')
    parser.add_argument('--weights', nargs='+', type=str, 
                       default='weights/End-to-end.pth', 
                       help='模型权重文件路径 (如: weights/model.pth)')
    parser.add_argument('--source', type=str, 
                       default='inference/images', 
                       help='输入源: 图片文件夹/视频文件/摄像头编号(0,1,2...)')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='推理图像尺寸 (像素)')
    parser.add_argument('--conf-thres', type=float, default=0.25, 
                       help='目标检测置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                       help='NMS的IOU阈值')
    parser.add_argument('--device', default='cpu', 
                       help='计算设备: cpu 或 cuda设备编号 (如: 0 或 0,1,2,3)')
    parser.add_argument('--save-dir', type=str, default='inference/output', 
                       help='结果保存目录')
    parser.add_argument('--augment', action='store_true', 
                       help='启用增强推理')
    parser.add_argument('--update', action='store_true', 
                       help='更新所有模型')
    
    opt = parser.parse_args()
    
    # 打印配置信息
    print("🚀 YOLOP车道线检测启动")
    print("=" * 50)
    print(f"权重文件: {opt.weights}")
    print(f"输入源:   {opt.source}")
    print(f"输出目录: {opt.save_dir}")
    print(f"计算设备: {opt.device}")
    print(f"图像尺寸: {opt.img_size}")
    print("=" * 50)
    print("功能特色:")
    print("✅ 实线显示: 红色")
    print("✅ 虚线显示: 黄色") 
    print("✅ 可行驶区域: 绿色")
    print("✅ 实时按 'q' 键退出")
    print("=" * 50)
    
    with torch.no_grad():
        detect(cfg, opt)