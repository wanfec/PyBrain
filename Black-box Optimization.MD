#Black-box Optimization
##1、连续函数优化
定义一个目标函数

	>>> def objF(x): return sum(x**2)
an initial guess ：

	>>> x0 =array([2.1, -1])
	
初始化一个优化算法（CMAES）：

	>>> from pybrain.optimization importCMAES
	>>> l =CMAES(objF,x0)
	
所有的优化算法都默认是最大化目标函数，这里需要最小化：

	>>> l.minimize = True
	
优化终止的条件是算法特定的，但是也可以人为的定义：

- maximal number of evaluations
- maximal number of learning steps
- reaching a desired value

**

	>>> l.maxEvaluations = 200
	
	
用learn()方法开始优化，直到达到优化终止的条件，返回最好的结果以及目标函数：

	>>> l.learn()
	(array([ -1.59778097e-05,  -1.14434779e-03]), 1.3097871509722648e-06)
	
###2、使用Evolvable进行一般优化
定义一个Evolvable的子类，要求实现：

- a copy() operator,
- a method for generating random other points: randomize(),
- mutate(), an operator that does a small step in search space, according to some distance metric,
- (optionally) a crossover() operator that produces some combination with other evolvables of the same class.

优化算法会初始化一个该类的实例，目标函数可以评估这个实例
例子（含有一个范围约束的变量，以及变化的步长）：

	>>> from random importrandom
	>>> from pybrain.structure.evolvables.evolvable importEvolvable
	>>> class SimpleEvo(Evolvable):
	...     def __init__(self,x): self.x = max(0, min(x, 10))
	...     def mutate(self):      self.x = max(0, min(self.x +random() - 0.3, 10))
	...     def copy(self):        returnSimpleEvo(self.x)
	...     def randomize(self):   self.x = 10*random()
	...     def __repr__(self):    return '<-%.2f->'+str(self.x)
	
可以用HillClimber进行优化

	>>> from pybrain.optimization importHillClimber
	>>> x0 =SimpleEvo(1.2)
	>>> l =HillClimber(lambdax:x.x,x0,maxEvaluations = 50)
	>>> l.learn()
	(<-10.00->, 10)

3、增强学习中的优化
