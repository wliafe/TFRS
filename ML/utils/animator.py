from matplotlib import pyplot as plt


class Animator:
    def __init__(self, legend, x_label=None, y_label=None, x_lim=None, y_lim=None, fig_size=None):
        self.legend, self.x_label, self.y_label, self.x_lim, self.y_lim = legend, x_label, y_label, x_lim, y_lim
        if fig_size is None:
            self.fig, self.axes = plt.subplots()
        else:
            self.fig, self.axes = plt.subplots(figsize=fig_size)
        self.axes.grid()
        self.X = [[] for _ in range(len(legend))]
        self.Y = [[] for _ in range(len(legend))]
        self.fmts = ['-', 'm--', 'g-.', 'r:']
        self.lines = []
        for i in range(len(legend)):
            liner, = self.axes.plot([], [], self.fmts[i])
            self.lines.append(liner)
        plt.show()

    def set_axes(self):
        self.axes.legend(self.legend)
        if self.x_label is not None:
            self.axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            self.axes.set_ylabel(self.y_label)
        if self.x_lim is not None:
            self.axes.set_xlim(self.x_lim)
        if self.y_lim is not None:
            self.axes.set_ylim(self.y_lim)

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        if not hasattr(x, "__len__"):
            x = [x] * len(y)
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append([a])
                self.Y[i].append([b])
        for x, y, liner in zip(self.X, self.Y, self.lines):
            liner.set_data(x, y)
        self.set_axes()
        self.axes.relim()
        self.axes.autoscale_view()
        self.fig.canvas.draw()
