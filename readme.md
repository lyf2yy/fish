1.数据集选取：我们使用了作者开源的带有标签的数据，只有fish一种标签，没有区分red fish和black fish。 数据一共包含954张图片，我们按照7：2：1的比例将数据集划分为训练数据、验证数据和测试数据，各有668、190、96张图片。如下图所示，使用yolo格式的box将fish标记，有一些数据中很多条鱼聚集在一起，还有一些数据中一些鱼很难被发现，被波浪和石头所遮挡，这给我们的任务带来很大的挑战。

<img src="/home/liujinhao/Pictures/Screenshot from 2024-04-12 12-04-20.png" style="zoom:50%;" />

<img src="/home/liujinhao/Pictures/Screenshot from 2024-04-12 12-04-38.png" style="zoom:50%;" />



2.模型选取：
我们选择使用YoloV8目标检测算法来进行实验，YoloV8算法具体结构如下图所示，改进了网络的结构，提出了c2f结构和SPPF结构，在加快检测速度的同时提升了检测的性能。而且YoloV8中使用解偶检测头，分别计算分类损失和box回归损失，避免了两者之间的干扰，进一步提升了检测的性能。


图片yolov8.png

<img src="/home/liujinhao/Pictures/yolov8.png" style="zoom:50%;" />

3.实验结果


实验结果如下图所示：

results.png

<img src="/home/liujinhao/Pictures/results.png" style="zoom:50%;" />

从图中可以看到在训练过程中box损失函数分类、损失函数以及dfl损失函数都随着训练epoch的增加逐渐减小，最终稳定，而且在验证数据集上损失函数也在第100个epoch后逐渐稳定。验证数据集中的dfl甚至在第100个epoch后逐渐增大，说明发生了过拟合。可以从图中看到precision稳定在0.4左右，recall稳定在0.33左右，AP50 稳定在0.28左右，AP50大概只有0.09. 在测试数据集上AP50也达到了0.26，AP50达到了0.09。
检测结果如下图所示，可以看到大部分的fish都可以检测到，但是box有点偏差，而且最左侧的fish被漏掉了。

图片

<img src="/home/liujinhao/Pictures/Screenshot from 2024-04-12 16-34-26.png" style="zoom:50%;" />

<img src="/home/liujinhao/Pictures/Screenshot from 2024-04-12 16-35-09.png" style="zoom:50%;" />

引用：
https://blog.csdn.net/lzy2766/article/details/132304440