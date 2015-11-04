PyBrain以神经网络为核心，所有的训练方法都以神经网络为一个实例(module)。

#Building Networks with Modules and Connections：
###1.前馈网络：
声明一个前馈网络n

	>>> from pybrain.structure importFeedForwardNetwork
	>>> n =FeedForwardNetwork()
	
构建输入层、输出层、隐藏层，并将其加入网络n

	>>> from pybrain.structure importLinearLayer,SigmoidLayer
	>>> inLayer =LinearLayer(2)
	>>> hiddenLayer =SigmoidLayer(3)
	>>> outLayer =LinearLayer(1)
	
	>>> n.addInputModule(inLayer)
	>>> n.addModule(hiddenLayer)
	>>> n.addOutputModule(outLayer)
	
声明层与层之间的连接方式(FullConnection)，并将其加入网络n

	>>> from pybrain.structure importFullConnection
	>>> in_to_hidden =FullConnection(inLayer,hiddenLayer)
	>>> hidden_to_out =FullConnection(hiddenLayer,outLayer)
	
	>>> n.addConnection(in_to_hidden)
	>>> n.addConnection(hidden_to_out)
	
所有设定完成之后，使网络可用：

	>>> n.sortModules()

###2、查看网络：

显示网络的结构

	>>> print n
	FeedForwardNetwork-6
	Modules:
	 [<LinearLayer 'LinearLayer-3'>, <SigmoidLayer 'SigmoidLayer-7'>, <LinearLayer 'LinearLayer-8'>]
	Connections:
	 [<FullConnection 'FullConnection-4': 'LinearLayer-3' -> 'SigmoidLayer-7'>, <FullConnection 'FullConnection-5': 'SigmoidLayer-7' -> 'LinearLayer-8'>]
	 
使用网络

	>>> n.activate([1, 2])
	array([-0.11302355])
	查看网络的参数（权值）
	>>> in_to_hidden.params
	array([ 1.37751406,  1.39320901, -0.24052686, -0.67970042, -0.5999425 , -1.27774679])
	>>> hidden_to_out.params
	array([-0.32156782,  1.09338421,  0.48784924])
	或
	>>> n.params
	array([ 1.37751406,  1.39320901, -0.24052686, -0.67970042, -0.5999425 ,	
	     -1.27774679, -0.32156782,  1.09338421,  0.48784924])
	     
给网络结构命名：

	>>> LinearLayer(2)
	<LinearLayer 'LinearLayer-11'>
	>>> LinearLayer(2,name="foo")
	<LinearLayer 'foo'>
	
**Recurrence networks**（In order to allow recurrency, networks have to be able to “look back in time”.）  
To create a recurrent network, just do as with feedforward networks but use the appropriate class:

	>>> from pybrain.structure importRecurrentNetwork
	>>> n =RecurrentNetwork()
	
We will quickly build up a network that is the same as in the example above:

	>>> n.addInputModule(LinearLayer(2,name='in'))
	>>> n.addModule(SigmoidLayer(3,name='hidden'))
	>>> n.addOutputModule(LinearLayer(1,name='out'))
	>>> n.addConnection(FullConnection(n['in'],n['hidden'],name='c1'))
	>>> n.addConnection(FullConnection(n['hidden'],n['out'],name='c2'))
	
The RecurrentNetwork class has one additional method, `.addRecurrentConnection()`, which looks back in time one timestep. We can add one from the hidden to the hidden layer:

	>>> n.addRecurrentConnection(FullConnection(n['hidden'],n['hidden'],name='c3'))
	
If we now activate the network, we will get different outputs each time:

	>>> n.sortModules()
	>>> n.activate((2, 2))
	array([-0.1959887])
	>>> n.activate((2, 2))
	array([-0.19623716])
	>>> n.activate((2, 2))
	array([-0.19675801])
	
Of course, we can clear the history of the network. This can be done by calling the reset method:

	>>> n.reset()
	>>> n.activate((2, 2))
	array([-0.1959887])
	>>> n.activate((2, 2))
	array([-0.19623716])
	>>> n.activate((2, 2))
	array([-0.19675801])
	
After the call to `.reset()`, we are getting the same outputs as just after the objects creation.
