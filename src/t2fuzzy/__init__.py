import numpy as np
from matplotlib import pyplot as plt

# Source: https://www.mathworks.com/help/fuzzy/predict-chaotic-time-series-using-type-2-fis.html

class IT2MembershipFcn:
    def __init__(self):
        # TODO - Handle other types besides gaussian MF's
        self.upper_mu: float = 0.5
        self.upper_sigma: float = 0.25
        self.lower_scale: float = 1.0
        self.lower_lag: float = 0.0

class T2OutputFcn:
    def __init__(self, c: np.ndarray | None = None):
        if c is None:
            c = np.zeros(0)
        self.c = c

    def __index__(self, idx: int) -> float:
        return self.c[idx]

    def __setitem__(self, idx: int, value: float):
        # Resize the 1D array to accommodate this index position
        if idx >= len(self.c):
            new_c = np.zeros(idx + 1)
            new_c[:len(self.c)] = self.c
            self.c = new_c

        self.c[idx] = value


class SugFisType2:
    def __init__(self):
        self.__inputs: list[IT2MembershipFcn] = []
        self.__outputs: list[T2OutputFcn] = [T2OutputFcn()]

    def inputs(self, idx: int):
        # Expand to include this index if needed.
        if idx >= len(self.__inputs):
            self.__inputs.extend([IT2MembershipFcn()] * (idx - len(self.__inputs) + 1))
        return self.__inputs[idx]

    def outputs(self, idx: int):
        if idx >= len(self.__outputs):
            self.__outputs.extend([T2OutputFcn()] * (idx - len(self.__outputs) + 1))
        return self.__outputs[idx]

    def add_inputs(self, num_mfs: int = 1):
        if num_mfs <= 0:
            raise ValueError("Number of membership functions must be positive")
        for idx in range(num_mfs):
            self.__inputs.append(IT2MembershipFcn())

    def add_outputs(self, range_x: list[float], num_mfs: int = 1):
        if num_mfs <= 0:
            raise ValueError("Number of membership functions must be positive")
        if len(range_x) != 2:
            raise ValueError("Range_x must be a list of two elements")
        if range_x[0] >= range_x[1]:
            raise ValueError("Range_x must be in ascending order")
        for idx in range(num_mfs):
            # TODO - Specify the input-output mapping?
            self.__outputs.append(T2OutputFcn())


def mackey_glass(x0: float = 1.2, dt: float = 1, tau: int = 20, n_samples: int = 1200):
    # Compute the recurrence relation
    x_t = np.zeros(n_samples)
    t_t = dt*np.arange(n_samples)
    i_tau = int(tau/dt)
    for ij, t in enumerate(t_t):
        if ij < i_tau:
            x_t[ij] = 0.0
        elif ij == i_tau:
            x_t[ij] = x0
        else:
            x_dot =0.2*x_t[ij-i_tau]/(1+x_t[ij-i_tau]**10) - 0.1*x_t[ij-1]
            x_t[ij] = x_t[ij-1] + x_dot*dt

    return t_t[i_tau:], x_t[i_tau:]


def main():
    # Compute the raw data
    t, x = mackey_glass()
    # Plot the mackey-glass function
    plt.figure()
    plt.plot(t,x)
    plt.title("Mackey-Glass Chaotic Time Series")
    plt.xlabel("$t$ [s]")
    plt.ylabel("$x(t)$")
    plt.show()

    # Create the training and validation data
    d = 4
    input_data = np.zeros((1000,d))
    output_data = np.zeros((1000,1))
    for ij in range(100+d,1100+d-1):
        for k in range(d):
            input_data[ij-100-d,k] = x[ij-d+k]
        output_data[ij-100-d+1,:] = x[ij+1]

    # Use first 500 as training data and last 500 as validation data
    trn_x = input_data[:500,:]
    trn_y = output_data[:500,:]
    vld_x = input_data[500:,:]
    vld_y = output_data[500:,:]

    fis_in = SugFisType2()
    n_inputs = d
    n_input_mfs = 3
    for ij in range(n_inputs):
        for jk in range(n_input_mfs):
            fis_in.inputs(ij).upper_mu = 0.5 + 0.1*jk




if __name__ == "__main__":
    main()