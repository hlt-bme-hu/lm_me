#coding: utf8

import numpy
import theano
import theano.tensor as T

class Optimizer(object):
    def __init__(self, objective, *vars, minimize=True, constraints={}, **kwargs):
        self.sym_vars = tuple(vars)
        self.grads = tuple(map(lambda var: T.grad(objective, var, **kwargs), vars))
        self.minimize = minimize
        self.objective = objective

        self.constraints = {var: var if var not in constraints else \
                                constraints[var] for var in vars}

    def updates(self):
        update_list = []
        grad_steps = self.grad_steps()
        sign = -1 if self.minimize else 1
        for i in range(len(self.sym_vars)):
            update_list.append([self.shr_vars[i],
                                self.constraints[self.sym_vars[i]] + sign*grad_steps[i]])
        return update_list
    def grad_steps(self):
        pass
    def init(self, *initvals, **kwargs):
        if len(initvals) != len(self.sym_vars):
            raise ValueError("Length of symbolic variables ({0}) is not equal to "
                              "length of initializer list ({1})!".format(
                                len(initvals), len(self.sym_vars))
                            )
        if hasattr(self, "shr_vars"):
            if len(initvals) != len(self.shr_vars):
                raise ValueError("initializer and already initialized shared variables"
                        " differ in length!")
            for i in range(len(initvals)):
                self.shr_vars[i].set_value(initvals[i])
        else:
            self.shr_vars = tuple(map(theano.shared, initvals))
    def givens(self):
        return tuple(zip(self.sym_vars, self.shr_vars))
    def get_vars(self):
        updates = self.renormalize_updates()
        return theano.function([],
                            [update[1] for update in updates],
                            givens=self.givens(),
                            updates=updates)

    def renormalize_updates(self):
        if not hasattr(self, "shr_vars"):
            raise AttributeError("object has no attribute 'shr_vars'! "
                        "Maybe you forgot to initialize the " +  type(self).__name__)
        new_vals = [self.constraints[var] for var in self.sym_vars]
        return list(zip(self.shr_vars, new_vals))

    @staticmethod
    def numpy_cast(x):
        return eval("numpy." + theano.config.floatX)(x)

class GradientDescentOptimizer(Optimizer):
    """
    vanilla
    """
    def __init__(self, objective, *vars, minimize=True, eta=1.0, constraints={}, **kwargs):
        super().__init__(objective, *vars, minimize=minimize, constraints=constraints, **kwargs)
        self.eta = eta

    def grad_steps(self):
        return tuple(map(lambda x: self.eta*x, self.grads))

class AdagradOptimizer(Optimizer):
    def __init__(self, objective, *vars, minimize=True, eta=1.0, constraints={}, **kwargs):
        super().__init__(objective, *vars, minimize=minimize, constraints=constraints, **kwargs)
        self.eta1 = Optimizer.numpy_cast(1.0/eta**2 if eta > 0 else numpy.inf)

    def updates(self):
        update_list = super().updates()
        for i in range(len(self.sym_vars)):
            update_list.append([self.grad_sqs[i], self.grad_sqs[i] + self.grads[i]**2])
        return update_list

    def grad_steps(self):
        return tuple(self.grads[i]/T.sqrt(self.grad_sqs[i]) for i in range(len(self.sym_vars)))

    def init(self, *initvals, keep_sq=False, **kwargs):
        super().init(*initvals, **kwargs)
        if not hasattr(self, "grad_sqs") or not keep_sq:
            self.grad_sqs = tuple(map(lambda x: theano.shared(self.eta1*numpy.ones_like(x)), initvals))
        
def Hessian(objective, *Vars, **kwargs):
     return T.concatenate([
                T.concatenate([
                    T.jacobian(
                        T.grad(objective, var1, disconnected_inputs='ignore').reshape(
                            (T.prod(var1.shape),)
                            ),
                        var2, disconnected_inputs='ignore').reshape(
                            (T.prod(var1.shape), T.prod(var2.shape))
                            ) for var2 in Vars],
                axis=1) for var1 in Vars],
            axis=0)
