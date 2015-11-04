#Classification with Feed-Forward Neural Networks

###1.声明（分类）数据集，并添加数据

	alldata =ClassificationDataSet(2, 1,nb_classes=3)
	forn in xrange(400):
	    forklass in range(3):
	        input =multivariate_normal(means[klass],cov[klass])alldata.addSample(input,[klass])

###2、把数据集分为训练集和测试集

	tstdata,trndata =alldata.splitWithProportion( 0.25)
###3、输出类型编码
对于用神经网络解决分类的问题，最好将分类的类型编码，每个类型编码为一个神经元的输出。注意，该操作首先把原数据集target中的内容复制到class，产生新的target：如果之前的target为2，新的target则为[010]）

	trndata._convertToOneOfMany( )
	tstdata._convertToOneOfMany( )
	
###4、例子
用buildNetwork()快速创建一个前馈网络

	fnn =buildNetwork(trndata.indim, 5,trndata.outdim,outclass=SoftmaxLayer)
	
用反向传播算法训练

	trainer =BackpropTrainer(fnn,dataset=trndata,momentum=0.1,verbose=True,weightdecay=0.01)
	
可视化
	
	ticks =arange(-3.,6.,0.2)
	X,Y =meshgrid(ticks,ticks)
	# need column vectors in dataset, not arrays
	griddata =ClassificationDataSet(2,1,nb_classes=3)
	for i in xrange(X.size):
		griddata.addSample([X.ravel()[i],Y.ravel()[i]],[0])
	griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

开始训练
	
	for i in range(20):
     trainer.trainEpochs( 1 )  # 训练一代

    # 评估每次训练后网络的准确性
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \

          "  test error: %5.2f%%" % tstresult
     # 可视化曲面的数据
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)
	#绘制数据点和分界曲线
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class']==c)
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on

    draw()  # update the plot
	# keep showing the plot until user kills it.
	ioff()
	show()
