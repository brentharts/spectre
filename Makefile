default:
	./somos8n.py --test10

t2:
	./somos8n.py --test2

t3:
	./somos8n.py --test3

t4:
	./somos8n.py --test4

t5:
	./somos8n.py --test5

t6:
	./somos8n.py --test6

t7:
	./somos8n.py --test7

t8:
	./somos8n.py --test8

t9:
	./somos8n.py --test9


a:
	blender --python einstein_fibration.py -- --iter2 --b=0.5

b:
	blender --python einstein_fibration.py -- --iter2 --b=1

c:
	blender --python einstein_fibration.py -- --iter2 --b=1.5

d:
	blender --python einstein_fibration.py -- --iter2 --b=3
