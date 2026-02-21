###########################################################################
#               Physics-Informed Machine Learning                         #
#                             WS 2025                                     #
#                                                                         #
#                           Exercise 4                                    #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import sys
import numpy as np

import matplotlib

try:
    matplotlib.use("TkAgg", force=True)
except Exception as e:
    print("Could not use TkAgg backend. GUI version requires TkAgg.")
    print("Error:", e)
    sys.exit(1)

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from functions import *
from optimizers import *

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class TorchOptimizerWrapper:
    """
    Wraps torch.optim.* so we can use it as a drop-in replacement for
    the exercise optimizers. It takes a NumPy-based gradient function
    and manually feeds gradients into torch.
    """
    def __init__(self, optimizer_name, point, grad_fn, lr, beta1=0.9, beta2=0.999):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available, cannot use reference optimizers.")

        self.grad_fn = grad_fn
        self.device = torch.device("cpu")

        init = torch.tensor(point, dtype=torch.float32, device=self.device)
        self.param = torch.nn.Parameter(init)

        if optimizer_name == "Gradient_descent":
            self.opt = torch.optim.SGD([self.param], lr=lr)
        elif optimizer_name == "RMSProp":
            self.opt = torch.optim.RMSprop([self.param], lr=lr, alpha=beta2)
        elif optimizer_name == "ADAM":
            self.opt = torch.optim.Adam([self.param], lr=lr, betas=(beta1, beta2))
        else:
            raise ValueError(f"Unsupported optimizer_name={optimizer_name!r} for TorchOptimizerWrapper")

    def step(self, point):
        """
        Same interface as the student optimizers: takes a NumPy point,
        returns the next NumPy point.
        """
        with torch.no_grad():
            self.param.data[:] = torch.tensor(point, dtype=self.param.dtype, device=self.device)

        grad_np = np.array(self.grad_fn(point), dtype=np.float32)
        grad_t = torch.from_numpy(grad_np).to(self.device)

        self.opt.zero_grad()
        self.param.grad = grad_t
        self.opt.step()

        return self.param.detach().cpu().numpy()


class OptimizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Optimization Demo (lighter)")

        self.grid_res = 60
        self.steps_per_frame = 3
        self.surface_update_every = 5
        self.interval_ms = 80

        self.fun_name = "Convex_Func"
        self.optimizer_name = "ADAM"
        self.lr = 2.5e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_iter = 1000

        self.fun_cls = None
        self.optimizer = None

        self.point = None
        self.xlist = []
        self.ylist = []
        self.zlist = []
        self.iter = 0
        self.running = False

        self.X = None
        self.Y = None
        self.Z = None

        self.line2d = None
        self.point2d = None
        self.text2d = None
        self.iter_line = None

        self.line3d = None
        self.point3d = None
        self.text3d = None

        self._build_gui()

        self.master.bind("<Return>", lambda event: self.start_optimization())

    def _build_gui(self):
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        plot_frame = ttk.Frame(self.master)
        plot_frame.pack(side="right", fill="both", expand=True)

        functions = ["Convex_Func", "Beale_Func", "Himmelblaus_Func",
                     "Rosenbrock_Func", "Ackley_Func"]
        optimizers = ["Gradient_descent", "Newton_Method_GD",
                      "RMSProp", "ADAM"]

        self.fun_var = tk.StringVar(value=self.fun_name)
        self.optimizer_var = tk.StringVar(value=self.optimizer_name)
        self.lr_var = tk.StringVar(value=str(self.lr))
        self.beta1_var = tk.StringVar(value=str(self.beta1))
        self.beta2_var = tk.StringVar(value=str(self.beta2))
        self.max_iter_var = tk.StringVar(value=str(self.max_iter))

        self.show_3d_var = tk.BooleanVar(value=True)

        self.impl_var = tk.StringVar(value="Yours")

        padding = {"padx": 5, "pady": 3}

        # row 0: function
        ttk.Label(control_frame, text="Objective function:") \
            .grid(row=0, column=0, sticky="w", **padding)
        fun_combo = ttk.Combobox(
            control_frame, textvariable=self.fun_var,
            values=functions, state="readonly"
        )
        fun_combo.grid(row=0, column=1, **padding)

        # row 1: optimizer
        ttk.Label(control_frame, text="Optimizer:") \
            .grid(row=1, column=0, sticky="w", **padding)
        opt_combo = ttk.Combobox(
            control_frame, textvariable=self.optimizer_var,
            values=optimizers, state="readonly"
        )
        opt_combo.grid(row=1, column=1, **padding)

        # row 2: learning rate
        ttk.Label(control_frame, text="Learning rate (alpha):") \
            .grid(row=2, column=0, sticky="w", **padding)
        ttk.Entry(control_frame, textvariable=self.lr_var, width=10) \
            .grid(row=2, column=1, sticky="w", **padding)

        # row 3: beta1
        ttk.Label(control_frame, text="beta1:") \
            .grid(row=3, column=0, sticky="w", **padding)
        ttk.Entry(control_frame, textvariable=self.beta1_var, width=10) \
            .grid(row=3, column=1, sticky="w", **padding)

        # row 4: beta2
        ttk.Label(control_frame, text="beta2:") \
            .grid(row=4, column=0, sticky="w", **padding)
        ttk.Entry(control_frame, textvariable=self.beta2_var, width=10) \
            .grid(row=4, column=1, sticky="w", **padding)

        # row 5: max_iter
        ttk.Label(control_frame, text="Max iterations:") \
            .grid(row=5, column=0, sticky="w", **padding)
        ttk.Entry(control_frame, textvariable=self.max_iter_var, width=10) \
            .grid(row=5, column=1, sticky="w", **padding)

        # row 6: 3D toggle
        ttk.Checkbutton(
            control_frame,
            text="Enable 3D surface (slower)",
            variable=self.show_3d_var
        ).grid(row=6, column=0, columnspan=2, sticky="w", **padding)

        # row 7: implementation mode
        ttk.Label(control_frame, text="Implementation:") \
            .grid(row=7, column=0, sticky="w", **padding)
        impl_combo = ttk.Combobox(
            control_frame,
            textvariable=self.impl_var,
            values=["Yours", "Reference (PyTorch)"],
            state="readonly"
        )
        impl_combo.grid(row=7, column=1, **padding)

        # row 8–9: buttons
        start_button = ttk.Button(
            control_frame, text="Start / Restart", command=self.start_optimization
        )
        start_button.grid(row=8, column=0, columnspan=2,
                          sticky="ew", pady=(10, 3))

        stop_button = ttk.Button(
            control_frame, text="Stop", command=self.stop_optimization
        )
        stop_button.grid(row=9, column=0, columnspan=2,
                         sticky="ew", pady=(0, 3))

        # plots
        notebook = ttk.Notebook(plot_frame)
        notebook.pack(fill="both", expand=True)

        frame2d = ttk.Frame(notebook)
        frame3d = ttk.Frame(notebook)
        notebook.add(frame2d, text="2D: Contour + Value")
        notebook.add(frame3d, text="3D surface")

        self.fig2d = Figure(figsize=(6, 4), dpi=100)
        self.ax_contour = self.fig2d.add_subplot(1, 2, 1)
        self.ax_loss = self.fig2d.add_subplot(1, 2, 2)
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=frame2d)
        self.canvas2d.get_tk_widget().pack(fill="both", expand=True)

        self.fig3d = Figure(figsize=(6, 4), dpi=100)
        self.ax_surface = self.fig3d.add_subplot(1, 1, 1, projection="3d")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=frame3d)
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)

    def _read_parameters(self):
        try:
            self.fun_name = self.fun_var.get()
            self.optimizer_name = self.optimizer_var.get()
            self.lr = float(self.lr_var.get())
            self.beta1 = float(self.beta1_var.get())
            self.beta2 = float(self.beta2_var.get())
            self.max_iter = int(self.max_iter_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Please make sure lr, beta1, beta2, and max_iter are numeric."
            )
            return False
        return True

    def _init_problem(self):
        if self.fun_name == 'Convex_Func':
            x_s, x_e = -3.0, 3.0
            y_s, y_e = -3.0, 3.0
            point = np.array([-2.5, 2.5])
        elif self.fun_name == 'Beale_Func':
            x_s, x_e = -4.0, 4.0
            y_s, y_e = -4.0, 4.0
            point = np.array([-3.8, 3.8])
        elif self.fun_name == 'Himmelblaus_Func':
            x_s, x_e = -8.0, 8.0
            y_s, y_e = -6.0, 6.0
            point = np.array([-4.75, 0.0])
        elif self.fun_name == 'Rosenbrock_Func':
            x_s, x_e = -5.0, 5.0
            y_s, y_e = -5.0, 5.0
            point = np.array([3.0, -2.0])
        else:  # Ackley
            x_s, x_e = -2.5, 2.5
            y_s, y_e = -2.5, 2.5
            point = np.array([0.35, 1.25])

        self.point = point
        self.xlist = [point[0]]
        self.ylist = [point[1]]

        # **coarser grid**
        y = np.linspace(y_s, y_e, self.grid_res)
        x = np.linspace(x_s, x_e, self.grid_res)
        self.X, self.Y = np.meshgrid(x, y)

        coords = np.concatenate(
            [np.expand_dims(self.X, 0), np.expand_dims(self.Y, 0)], axis=0
        )

        try:
            Z = np.array(self.fun_cls.eval(coords))
            z0 = float(np.array(self.fun_cls.eval(self.point)))
        except:
            messagebox.showerror(
                "Function not implemented",
                f"The objective '{self.fun_name}' is not fully implemented.\n"
                "Please complete its eval(...) in functions.py."
            )
            raise

        self.Z = np.squeeze(Z)
        self.zlist = [z0]
        self.iter = 0


    def _init_optimizer(self):
        """
        If implementation mode is 'Reference (PyTorch)' and torch is available,
        use TorchOptimizerWrapper for Gradient_descent / RMSProp / ADAM.
        Otherwise, fall back to the student implementations in optimizers.py.
        """

        impl_mode = getattr(self, "impl_var", None)
        impl_mode = impl_mode.get() if impl_mode is not None else "Student"

        self.optimizer = None

        try:
            _ = np.array(self.fun_cls.gradient(self.point))
            if self.optimizer_name == 'Newton_Method_GD':
                _ = np.array(self.fun_cls.hessian(self.point))
        except NotImplementedError:
            messagebox.showerror(
                "Gradient / Hessian not implemented",
                f"The gradient and/or Hessian for '{self.fun_name}' are not implemented.\n"
                "Please complete them in functions.py before running optimization."
            )
            raise

        if impl_mode == "Reference (PyTorch)":
            if not TORCH_AVAILABLE:
                messagebox.showerror(
                    "PyTorch not available",
                    "PyTorch is not installed; cannot use reference optimizers.\n"
                    "Please install torch or switch back to 'Yours'."
                )
            elif self.optimizer_name in ("Gradient_descent", "RMSProp", "ADAM"):
                self.optimizer = TorchOptimizerWrapper(
                    optimizer_name=self.optimizer_name,
                    point=self.point,
                    grad_fn=self.fun_cls.gradient,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
            else:
                messagebox.showinfo(
                    "Reference optimizer not available",
                    "The reference implementation only supports\n"
                    "Gradient_descent, RMSProp, and ADAM.\n"
                    "Falling back to your own Newton_Method_GD."
                )

        if self.optimizer is None:
            try:
                init_args = [self.fun_cls.gradient, self.lr]
                if self.optimizer_name == 'Newton_Method_GD':
                    init_args.append(self.fun_cls.hessian)
                    self.optimizer = Newton_Method_GD(*init_args)
                elif self.optimizer_name != 'Gradient_descent':
                    init_args.append(self.beta2)
                    if self.optimizer_name == 'ADAM':
                        init_args.append(self.beta1)
                        self.optimizer = ADAM(*init_args)
                    else:
                        self.optimizer = RMSProp(*init_args)
                else:
                    self.optimizer = Gradient_Descent(*init_args)
            except NameError:
                messagebox.showerror(
                    "Optimizer not implemented",
                    f"The optimizer '{self.optimizer_name}' is not defined in optimizers.py.\n"
                    "Please implement it before selecting 'Yours'."
                )
                raise


    def _setup_plots(self):
        self.ax_contour.clear()
        self.ax_contour.contour(self.X, self.Y, self.Z, 25, cmap='jet')
        self.ax_contour.set_title(self.fun_name)
        self.ax_contour.set_xlabel('x')
        self.ax_contour.set_ylabel('y')

        (self.line2d,) = self.ax_contour.plot([], [], 'r-', label=self.optimizer_name, lw=1.5)
        (self.point2d,) = self.ax_contour.plot([], [], '*', color='k', markersize=10)
        self.text2d = self.ax_contour.text(0.02, 0.02, '', transform=self.ax_contour.transAxes)
        self.ax_contour.legend(loc="upper right")

        self.ax_loss.clear()
        (self.iter_line,) = self.ax_loss.plot([], [])
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Function value')
        self.ax_loss.set_title('Function value vs iteration')
        self.ax_loss.set_xlim(0, max(1, self.max_iter))
        self.ax_loss.set_ylim(min(self.zlist) * 0.9,
                              max(self.zlist) * 1.1 if self.zlist[0] != 0 else 1.0)

        self.canvas2d.draw_idle()

        self.ax_surface.clear()
        if self.show_3d_var.get():
            self.ax_surface.set_xlabel('x (or parameter)')
            self.ax_surface.set_ylabel('y (or parameter)')
            self.ax_surface.set_zlabel('z (or loss)')
            self.ax_surface.set_title(self.fun_name)

            self.ax_surface.plot_surface(
                self.X, self.Y, self.Z,
                rstride=3, cstride=3, cmap='jet', alpha=0.5
            )
            (self.line3d,) = self.ax_surface.plot([], [], [], 'r-', label=self.optimizer_name, lw=1.5)
            (self.point3d,) = self.ax_surface.plot([], [], [], '*', color='k', markersize=10)
            self.text3d = self.ax_surface.text2D(
                0.02, 0.95, '', transform=self.ax_surface.transAxes
            )
            self.ax_surface.legend()
        else:
            self.ax_surface.text2D(
                0.5, 0.5,
                "3D disabled for speed",
                transform=self.ax_surface.transAxes,
                ha="center", va="center"
            )
            self.line3d = None
            self.point3d = None
            self.text3d = None

        self.canvas3d.draw_idle()

    def start_optimization(self):
        self.running = False

        if not self._read_parameters():
            return

        try:
            self.fun_cls = eval(self.fun_name)
        except NameError:
            messagebox.showerror(
                "Function not found",
                f"{self.fun_name} is not defined (check functions.py)."
            )
            return

        try:
            self._init_problem()
            self._init_optimizer()
        except NotImplementedError:
            self.running = False
            return

        self._setup_plots()
        self.running = True
        self.master.after(self.interval_ms, self._step_and_schedule)


    def stop_optimization(self):
        self.running = False

    def _step_and_schedule(self):
        if not self.running:
            return

        for _ in range(self.steps_per_frame):
            if not self.running:
                break
            if self.iter >= self.max_iter:
                self.running = False
                break

            try:
                self.point = self.optimizer.step(self.point)
                z = float(np.array(self.fun_cls.eval(self.point)))
            except NotImplementedError:
                impl_mode = self.impl_var.get()
                if impl_mode == "Reference (PyTorch)":
                    msg = (
                        "The selected objective is not fully implemented "
                        "(eval/gradient/Hessian) in functions.py.\n"
                        "The reference optimizer cannot run without these."
                    )
                else:
                    msg = (
                        f"It looks like your optimizer '{self.optimizer_name}' "
                        "is still marked as 'not implemented', or the objective "
                        "eval(...) is still a stub.\n"
                        "Please complete the code in optimizers.py / functions.py."
                    )
                messagebox.showerror("Method not implemented", msg)
                self.running = False
                return

            self.xlist.append(self.point[0])
            self.ylist.append(self.point[1])
            self.zlist.append(z)
            self.iter += 1

            if z < 1e-8:
                self.running = False
                break

        if len(self.zlist) == 0:
            return

        self.line2d.set_data(self.xlist, self.ylist)
        self.point2d.set_data([self.point[0]], [self.point[1]])
        self.text2d.set_text(f"Iter: {self.iter}  f={self.zlist[-1]:.3e}")

        iters = np.arange(len(self.zlist))
        self.iter_line.set_data(iters, self.zlist)
        self.ax_loss.set_xlim(0, max(1, len(self.zlist) - 1))
        ymin = min(self.zlist)
        ymax = max(self.zlist)
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        self.ax_loss.set_ylim(ymin, ymax)

        self.canvas2d.draw_idle()

        if self.show_3d_var.get() and self.line3d is not None:
            if (self.iter % self.surface_update_every == 0) or (not self.running):
                self.line3d.set_data(self.xlist, self.ylist)
                self.line3d.set_3d_properties(self.zlist)
                self.point3d.set_data([self.point[0]], [self.point[1]])
                self.point3d.set_3d_properties([self.zlist[-1]])
                self.text3d.set_text(f"Iter: {self.iter}  f={self.zlist[-1]:.3e}")
                self.canvas3d.draw_idle()

        if self.running:
            self.master.after(self.interval_ms, self._step_and_schedule)


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
