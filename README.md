# kaijutuoersuo
本分支不采用OCR，使用cv2根据预定义数字匹配识别

开局托儿所自动运行策略_windows and mac

> 在1080p mac与windows上测试通过，其他分辨率有待测试

## 使用说明
* 可选
0. 安装miniconda，以[Miniconda Windows](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)为例，按照引导安装

1、安装python后，**进入目录**后，pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

2、使用电脑微信打开“开局托儿所”小程序，在游戏界面内点击“开始挑战”后，等页面稳定倒计时后，运行 python main.py，按照提示运行


*破纪录后需要领取东西，可能导致软件退出

## 游戏分析

这个游戏的具体生成策略没有仔细研究过，只能确定它不是完全随机的，完全随机数的1000次测试结果如下图1，大部分分数集中在120左右。而游戏实际运行677次的测试结果如图2，可以看到期望大概100多，低于随机。测试说明游戏的矩阵生成故意增加了难度。

图中不同颜色代表不同的游戏策略，详见以下策略解析。

![随机生成矩阵](pic/random.png)

![真实测试](pic/real.png)

## 策略解析

这个算法题的难点在于多步骤，如果是单步的求解什么，就类似于《最大子列和》问题，只要一步动态规划就可以求解。多步消除的上一步会影响下一步，而且这个矩阵还挺大的，就有点围棋那个意思了（或许可以train一个RL来求解）。

综上，在此只提出一些简单的策略与测试。先说结论，根据数据分析，较少数字的消除优先级高，例如[1,9]的消除优先级大于[1,2,2,5]；横向的消除优先级高于竖向；决定分数上限的是运气，决定分数高低的是策略，例如以下151分的局面，6种策略能达到的分数就有较大区别，最低只有115，最高有151。具体算法见main.py。

![策略影响结果](pic/score.jpg)

Thanks to https://github.com/GlintFreedom/CVGameScript


