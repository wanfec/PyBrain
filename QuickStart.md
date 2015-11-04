#Building a Network：


1.快速建立一个神经网络（2个输入层，1个输出层，3个隐藏层）：

	>>>from pybrain.tools.shortcuts importbuildNetwork`
	>>> net =buildNetwork(2, 3, 1)`

2.使用神经网络（输入为[2,1]）
	
	>>> net.activate([2, 1])
	array([-0.98646726])
	
3.检查网络结构

	>>> net['in']
	<LinearLayer 'in'>
	>>> net['hidden0']
	<SigmoidLayer 'hidden0'>
	>>> net['out']
	<LinearLayer 'out'>
	
4、使用更复杂的网络

隐藏层为TanhLayer：

	>>> from pybrain.structure importTanhLayer
	>>> net =buildNetwork(2, 3, 1,hiddenclass=TanhLayer)
	>>> net['hidden0']
	<TanhLayer 'hidden0'>
	
输出层为SoftmaxLayer

	>>> from pybrain.structure importSoftmaxLayer
	>>> net =buildNetwork(2, 3, 2,hiddenclass=TanhLayer,outclass=SoftmaxLayer)
	>>> net.activate((2, 3))
	array([ 0.6656323,  0.3343677])
	
使用bias:

	>>> net =buildNetwork(2, 3, 1,bias=True)
	>>> net['bias']
	<BiasUnit 'bias'>
	
**PyBrain只能构建前向网络**

--

#Building a DataSet：

1、自定义数据集（数据集有2个输入，1个输出）

	>>> from pybrain.datasets import SupervisedDataSet
	>>> ds =SupervisedDataSet(2, 1)

2、向数据集中添加样本

	>>> ds.addSample((0, 0),(0,))
	>>> ds.addSample((0, 1),(1,))
	>>> ds.addSample((1, 0),(1,))
	>>> ds.addSample((1, 1),(0,))

3、查看数据集  
数据集的大小：

	>>> len(ds)

for循环访问数据集：

	>>> for inpt,target in ds:
	…   print inpt,target
	...
	[ 0.  0.] [ 0.]
	[ 0.  1.] [ 1.]
	[ 1.  0.] [ 1.]
	
	[ 1.  1.] [ 0.]
用array的方式访问input and target field 

	>>> ds['input']
	array([[ 0.,  0.],
	       [ 0.,  1.],
	       [ 1.,  0.],
	       [ 1.,  1.]])
	>>> ds['target']
	array([[ 0.],
	       [ 1.],
	       [ 1.],	
	       [ 0.]])

清空数据集

	>>> ds.clear()
	>>> ds['input']
	array([], shape=(0, 2), dtype=float64)
	>>> ds['target']
	array([], shape=(0, 1), dtype=float64)
	
--	

#Training your Network on your Dataset：

1、用反向传播算法进行训练：

	>>> from pybrain.supervised.trainers import BackpropTrainer
	
2、定义一个trainer，用当前的网络结构(net)、数据集(ds)进行训练：

	>>> net =buildNetwork(2, 3, 	1,bias=True,hiddenclass=TanhLayer)
	>>> trainer =BackpropTrainer(net,ds)
	
3、训练一次，返回训练误差：

	>>> trainer.train()
	0.31516384514375834
	
4、训练直到收敛，返回一个元组，包含每次的训练误差

	>>> trainer.trainUntilConvergence()
	...
