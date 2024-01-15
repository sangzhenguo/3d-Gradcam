# 3d-Gradcam
在中文互联网中并没有找到3d-cam可视化的资料，monai中的3d可视化不适合我的需求，于是自己修改了3d-Gradcam。初始官方代码是https://github.com/jacobgil/pytorch-grad-cam/tree/master
## 主要修改了：
- grad_cam.py中的get_cam_weights，将axis=(2, 3)修改为axis=(3, 4)，适应3d的维度
- base_cam.py中的BaseCAM基类中的get_cam_image方法，指定输出weights这个3d数组中的特定2d维度子数组。
- 由于自己特定层的输出并不是一个单一数组，还修改了ActivationsAndGradients中的output
- 可以根据自己的数据特点修改image.py中的show_cam_on_image
- test.py

## test.py
是利用gradcam调用unet中特定层输出可视化图的文件
GradCAM也要指定输出层
