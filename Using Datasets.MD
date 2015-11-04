#Using Datasets
如果DS是一个DataSet的实例

	inp =DS['input']
	
会返回input field

	inp[0,:]
	
会返回第一个输入向量（对target也同样适用）

iterate over a dataset：

	for inp,targ in DS:
	  ...
向DataSet中添加样本（逐个样本添加or直接从一个向量中添加）：

	for inp,tart in samples:
	     DS.appendLinked(inp,targ)
     
	# or alternatively, with  ia  and  ta  being arrays:

	assert(ia.shape[0] ==ta.shape[0])
	DS.setField('input',ia)
	DS.setField('target',ta)
向数据集中添加linked or unlinked data

	DS.addField('myfield')
	DS.setField('myfield',myarray)
	DS.linkFields('input','target','myfield') # must provide complete list here

快速（随机）产生训练集和测试集

	>>> len(DS)
	100
	>>> TrainDS,TestDS =DS.splitWithProportion(0.8)
	>>> len(TrainDS), len(TestDS)
	(80, 20)
	
1.监督学习数据集

	>>> from pybrain.datasets importSupervisedDataSet
	>>> DS =SupervisedDataSet( 3, 2)
	>>> DS.appendLinked([1,2,3],[4,5])
	>>> len(DS)
	1
	>>> DS['input']
	array([[ 1.,  2.,  3.]])
	
2.监督序列学习数据集（把数据集分为长度不同的序列，分别使用）
3、分类训练数据集
初始化：

	DS =ClassificationDataSet(inputdim,nb_classes=2,class_labels=['Fish','Chips'])
label是可选的，target的维度必须为1，target是一个从0开始的数字，如果不知道有多少个类型：

	>>> DS =ClassificationDataSet(2,class_labels=['Urd', 'Verdandi', 'Skuld'])
	>>> DS.appendLinked([ 0.1, 0.5],[0])
	>>> DS.appendLinked([ 1.2, 1.2],[1])
	>>> DS.appendLinked([ 1.4, 1.6],[1])
	>>> DS.appendLinked([ 1.6, 1.8],[1])
	>>> DS.appendLinked([ 0.10, 0.80],[2])
	>>> DS.appendLinked([ 0.20, 0.90],[2])

	>>> DS.calculateStatistics()
	{0: 1, 1: 3, 2: 2}
	>>> printDS.classHist
	{0: 1, 1: 3, 2: 2}
	>>> printDS.nClasses
	3
	>>> printDS.getClass(1)
	Verdandi
	>>> printDS.getField('target').transpose()
	[[0 1 1 1 2 2]]
	
编码分类的类型，把target中的内容复制到class，产生新的target；例如，之前的target为2，新的target则为[010]

	>>> DS._convertToOneOfMany(bounds=[0, 1])
	>>> printDS.getField('target')
	[[1 0 0]
	 [0 1 0]
	 [0 1 0]
	 [0 1 0]
	 [0 0 1]
	 [0 0 1]]
	>>> printDS.getField('class').transpose()
	[[0 1 1 1 2 2]]
	>>> DS._convertToClassNb()
	>>> printDS.getField('target').transpose()
	[[0 1 1 1 2 2]]
	
4、带权值的监督学习数据集
