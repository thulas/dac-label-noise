
"""
A simple PID controller
"""

#PID time constant
dt = 0.01

class pid_controller(object):

    def __init__(self, set_point, 
        k_p=0.1, k_i=0.1,k_d = 0.05,
        limits=(None, None)):

        self.set_point = set_point
        self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
        self.min_output, self.max_output = limits
        self.pv_last = None

        self.proportional = 0.
        self.integral = 0.
        self.derivative = 0.


    def clamp(self, value):
        if self.max_output is not None and value > self.max_output:
            return self.max_output
        elif self.min_output is not None and value < self.min_output:
            return self.min_output
        else:
            return value



    def __call__(self, pv):
        #error
        error = self.set_point - pv
        if self.pv_last is not None:
            d_pv = pv - self.pv_last
        else:
            d_pv = 0.
        
        #proportional term
        self.proportional = self.k_p * error

        #integral term
        self.integral += self.k_i * error *dt
        #below to prevent integral windup
        self.integral = self.clamp(self.integral)

        #derivative term
        self.derivative = -self.k_d * d_pv / dt

        #output
        output = self.proportional + self.integral + self.derivative
        output = self.clamp(output)

        self.pv_last = pv
        return output


    def tunings(self):
        return self.k_p, self.k_i, self.k_d


    def components(self):
        return self.proportional, self.integral, self.derivative

