import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

from virocon.utils import calculate_design_conditions

__all__ = ["plot_marginal_quantiles", "plot_dependence_functions",
           "plot_2D_isodensity", "plot_2D_contour"]


 # TODO move to utility as it is also used in contours.py
def get_default_model_description(n_dim):  
    """
       
       TODO: Beschreibung
       
    """
        
    model_desc = {"names" : [f"Variable {dim+1}" for dim in range(n_dim)],
                  "symbols" : [f"V{dim+1}" for dim in range(n_dim)],
                  "units" : ["arb. unit" for dim in range(n_dim)]
                  }
    return model_desc


def plot_marginal_quantiles(model, sample, model_desc=None, axes=None):
    """
       
       TODO: Beschreibung
       
    """
        
    sample = np.asarray(sample)
    n_dim = model.n_dim
    if model_desc is None:
        model_desc = get_default_model_description(n_dim)
    
    if axes is None:
        axes = []
        for i in range(n_dim):
            _, ax = plt.subplots()
            axes.append(ax)
            
    
    # probplot expects an object that has a ppf method, but we name it icdf
    # therefor we create a wrapper that maps the ppf to the icdf method
    class MarginalDistWrapper:
            
        """
           
        TODO: Beschreibung
           
        """
        
        
        def __init__(self, model, idx):
            self.model = model
            self.idx = idx
            
        def ppf(self, q):
            return self.model.marginal_icdf(q, self.idx)
        
    for dim in range(n_dim):
        dist_wrapper = MarginalDistWrapper(model, dim)
        ax = axes[dim]
        
        
        sts.probplot(sample[:, dim], dist=dist_wrapper, fit=False, plot=ax)
        ax.get_lines()[0].set_markerfacecolor("k")
        ax.get_lines()[0].set_markeredgecolor("k")
        ax.get_lines()[0].set_marker("x")
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color("#004488")
        name_and_unit = f"{model_desc['names'][dim].lower()} ({model_desc['units'][dim]})"
        ax.set_xlabel(f"Theoretical quantiles of {name_and_unit}")
        ax.set_ylabel(f"Ordered values of {name_and_unit}")
        ax.title.set_text("")
        
    return axes

def plot_dependence_functions(model, model_desc=None, par_rename=None, axes=None):
    """
       
    TODO: Beschreibung
       
    """
        
    n_dim = model.n_dim
    conditional_dist_idc = [dim for dim  in range(n_dim) 
                            if model.conditional_on[dim] is not None]
    
    if model_desc is None:
        model_desc = get_default_model_description(n_dim)
    
    if axes is None:
        n_axes = 0
        for dim in conditional_dist_idc:
            n_axes += len(model.distributions[dim].conditional_parameters)
        axes = []
        for i in range(n_axes):
            _, ax = plt.subplots()
            axes.append(ax)
            
    axes_counter = 0
    for dim in conditional_dist_idc:
        dist = model.distributions[dim]
        conditioning_values = dist.conditioning_values
        x = np.linspace(0, max(conditioning_values))
        cond_idx = model.conditional_on[dim]
        x_name = model_desc["names"][cond_idx]
        x_symbol = model_desc["symbols"][cond_idx]
        x_unit = model_desc["units"][cond_idx]
        x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
        for par_name, dep_func in dist.conditional_parameters.items():
            par_values = [par[par_name] for par in dist.parameters_per_interval]
            ax = axes[axes_counter]
            axes_counter += 1
            ax.scatter(conditioning_values, par_values, c="k", marker="x")
            ax.plot(x, dep_func(x), c="#004488")
            ax.set_xlabel(x_label)
            if par_name in par_rename:
                y_label = par_rename[par_name]
            else:
                y_label = par_name
            ax.set_ylabel(y_label)    
            
    return axes


def plot_2D_isodensity(model, sample, model_desc=None, swap_axis=False, ax=None):   
    """
       
       TODO: Beschreibung
       
    """
        

    n_dim = model.n_dim
    assert n_dim == 2
    
    if swap_axis:
        x_idx = 1
        y_idx = 0
    else:
        x_idx = 0
        y_idx = 1
        
    
    if model_desc is None:
        model_desc = get_default_model_description(n_dim)
        
    if ax is None:
            _, ax  = plt.subplots()
            
    sample = np.asarray(sample)
    ax.scatter(sample[:, x_idx], sample[:, y_idx], c="k", marker=".", alpha=0.3)
        
    x_lower = min(sample[:, 0])
    x_upper = max(sample[:, 0])
    y_lower = min(sample[:, 1])
    y_upper = max(sample[:, 1])
    x, y = np.linspace(((x_lower, y_lower)), (x_upper, y_upper)).T
    X, Y = np.meshgrid(x, y)
    grid_flat = np.c_[X.ravel(), Y.ravel()]
    f = model.pdf(grid_flat)
    Z = f.reshape(X.shape)
    
    if swap_axis:
        tmp = X
        X = Y
        Y = tmp
        
    ax.contour(X, Y, Z, colors="#BB5566")
    x_name = model_desc["names"][x_idx]
    x_symbol = model_desc["symbols"][x_idx]
    x_unit = model_desc["units"][x_idx]
    x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
    y_name = model_desc["names"][y_idx]
    y_symbol = model_desc["symbols"][y_idx]
    y_unit = model_desc["units"][y_idx]
    y_label = f"{y_name}," + r" $\it{" + f"{y_symbol}" + r"}$" + f" ({y_unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return ax



def plot_2D_contour(contour, sample=None, design_conditions=None, model_desc=None, swap_axis=False, ax=None):

    """ 
       TODO: Beschreibung
       
    """
    
    # design conditions can be True or array
    n_dim = 2
    if swap_axis:
        x_idx = 1
        y_idx = 0
    else:
        x_idx = 0
        y_idx = 1
        
        
    if model_desc is None:
        model_desc = get_default_model_description(n_dim)
    
    if ax is None:
        _, ax  = plt.subplots()
        
        
    if design_conditions:
        try: # if iterable assume it's already the design conditions
            iter(design_conditions)
        except: # if it is not an array we compute the default design_conditions
            design_conditions = calculate_design_conditions(contour, swap_axis=swap_axis)
            
        ax.scatter(design_conditions[:, 0], design_conditions[:, 1], c="#DDAA33", marker="x", 
            zorder=2.5)
    
    if sample is not None:
        sample = np.asarray(sample)
        ax.scatter(sample[:, x_idx], sample[:, y_idx], c="k", marker=".", alpha=0.3)
        
    
    coords = contour.coordinates
    x = coords[:, x_idx].tolist()
    x.append(x[0])
    y = coords[:, y_idx].tolist()
    y.append(y[0])
    ax.plot(x, y, c="#BB5566")
    
    x_name = model_desc["names"][x_idx]
    x_symbol = model_desc["symbols"][x_idx]
    x_unit = model_desc["units"][x_idx]
    x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
    y_name = model_desc["names"][y_idx]
    y_symbol = model_desc["symbols"][y_idx]
    y_unit = model_desc["units"][y_idx]
    y_label = f"{y_name}," + r" $\it{" + f"{y_symbol}" + r"}$" + f" ({y_unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return ax