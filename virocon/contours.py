import os
import warnings

import numpy as np
import scipy.stats as sts
import scipy.ndimage as ndi
import networkx as nx

from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors

from virocon._nsphere import NSphere
from virocon.plotting import get_default_model_description

__all__ = ["calculate_alpha", "save_contour_coordinates", "IFORMContour",
           "ISORMContour", "HighestDensityContour", "DirectSamplingContour"]

def calculate_alpha(state_duration, return_period):
    """
    Calculates the probability that an environmental contour is exceeded
    (exceedance probability). 
    
    The exceedance probability, α, corresponds to a certain recurrence or 
    return period, T, which describes the average time period between two 
    consecutive environmental states that exceed the contour . Note that 
    exceedance can be defined in various ways for environmental contours
    (Mackay and Haselsteiner, 2021) [1]_
    
    Parameters
    ---------- 
    state_duration : float
        Time period for which an environmental state is measured,
        expressed in hours :math:`(T_s)`.
    return_period : float
        Describes the average time period between two consecutive 
        environmental states that exceed a contour. In the univariate case the 
        contour is a threshold, x1.
    
        :math:`\\alpha= \\frac{T_s}{T_r * 365.25 * 24}` 
        
        :math:`F(x_1) =  P(X_1 \geq x_1)= \int_{- \infty}^{x_1} f(x) dx = 1- \\alpha`  
        
    Returns
    -------
    alpha : float
        The probability that an environmental contour is exceeded.
    
    References
    ----------
    .. [1] Mackay, E., & Haselsteiner, A. F. (2021). 
       Marginal and total exceedance probabilities of environmental contours. 
       Marine Structures, 75. https://doi.org/10.1016/j.marstruc.2020.102863

    """
    
    alpha = state_duration / (return_period * 365.25 * 24)
    return alpha


def sort_points_to_form_continuous_line(x, y, search_for_optimal_start=False):
    """
    Sorts contour points to form a a continous line / contour.
    
    This function simply sorts 2 dimensional points. The points are given 
    by x and y coordinates. This function is used to sort the coordinates 
    of 2 dimensional contours.

    Thanks to https://stackoverflow.com/a/37744549

    Parameters
    ----------
    x : array_like
        x-coordinate of contour points.
    y : array_like
        y-coordinate of contour points.
    search_for_optimal_start : boolean, optional
        If True, the algorithm also searches for the ideal starting node, see the
        stackoverflow link for more info. Defaults to False.

    Returns
    -------
    sorted_points : tuple of array_like floats
        The sorted points.
    """
    
    points = np.c_[x, y]
    clf = NearestNeighbors(n_neighbors=2).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    order = list(nx.dfs_preorder_nodes(T, 0))

    xx = x[order]
    yy = y[order]

    if search_for_optimal_start:
        paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
        mindist = np.inf
        minidx = 0

        for i in range(len(points)):
            p = paths[i]  # Order of nodes.
            ordered = points[p]  # Ordered nodes.
            # Find cost of that order by the sum of euclidean distances
            # between points (i) and (i + 1).
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = i

        opt_order = paths[minidx]

        xx = x[opt_order]
        yy = y[opt_order]

    return xx, yy


def save_contour_coordinates(contour, file_path, model_desc=None):
    """
    Saves the coordinates of the calculated contour. 
    Saves a .txt file to the given path.
    
    Parameters
    ----------
    contour : Contour
     The contour with the coordinates to save.
    file_path : string
     Indicates the path, where the contour coordinates are saved.
    model_desc : dictionary
     The description of the model. model_desc has the keys 'names', 'symbols' 
     and 'units'. Each value is a list of strings. For each dimension of the 
     model the strings describe the name, symbol or unit of that dimension, 
     respectively. This information is used as the header of the created file.
     Defaults to a dict with general descriptions.
     
    """
    
    root, ext = os.path.splitext(file_path)
    if not ext:
        file_path += ".txt"
    
    n_dim = contour.coordinates.shape[1]
    if model_desc is None:
        model_desc = get_default_model_description(n_dim)
        
    header = ";".join((f"{model_desc['names'][d]} ({model_desc['units'][d]})" 
                        for d in range(n_dim)))
    
    np.savetxt(file_path, contour.coordinates, fmt="%1.6f", delimiter=";", 
               header=header, comments="")


class Contour(ABC):
    """
      Abstract base class for contours.
      
      A contour implements a method to define multivariate extremes based on a
      joint probabilistic model of variables like significant wave height, 
      wind speed or spectral peak period.
      
      Contour curves or surfaces for more than two environmental parameters 
      give combination of environmental parameters which approximately 
      describe the various actions corresponding to the given exceedance 
      probability [1]_.
      
      
    References
    ----------
    .. [1] NORSOK standard N-003, Edition 2, September 2007. Actions and 
        action effects.


    """

    def __init__(self):
        try:
            _ = self.model
        except AttributeError:
            raise NotImplementedError(f"Can't instantiate abstract class {type(self).__name__} "
                                      "with abstract attribute model.")
        try:
            _ = self.alpha
        except AttributeError:
            raise NotImplementedError(f"Can't instantiate abstract class {type(self).__name__} "
                                      "with abstract attribute model.")

        self._compute()
        try:
            _ = self.coordinates
        except AttributeError:
            raise NotImplementedError(f"Can't instantiate abstract class {type(self).__name__} "
                                      "with abstract attribute coordinates.")

    @abstractmethod
    def _compute(self):
        """
        Compute the contours coordinates.

        Is automatically called in the __init__. 
        """
        pass


class IFORMContour(Contour):
    """
    Contour based on the inverse first-order reliability method.

    This method was proposed by Winterstein et. al (1993) [1]_

    Parameters
    ----------
    model :  MultivariateModel
        The model to be used to calculate the contour.
    alpha : float
        The exceedance probability. The probability that an observation falls 
        outside the environmental contour.
    n_points : int, optional
        Number of points on the contour. Defaults to 180.
        
    Attributes
    ----------
    coordinates : 
        Coordinates of the calculated contour.
    beta :
        Reliability index.  
    sphere_points : 
          Points of the sphere in U space [1]_ . 
    
    References
    ----------
    .. [1] Winterstein, S.R.; Ude, T.C.; Cornell, C.A.; Bjerager, P.; Haver, S. (1993)
        Environmental parameters  for extreme response: Inverse FORM with omission
        factors. ICOSSAR 93, Innsbruck, Austria. 


    """
  
    def __init__(self, model, alpha, n_points=180):
        self.model = model
        self.alpha = alpha
        self.n_points = n_points
        super().__init__()

    def _compute(self, ):
        """
        Calculates coordinates using IFORM.

        """
        n_dim = self.model.n_dim
        n_points = self.n_points
        distributions = self.model.distributions
        conditional_on = self.model.conditional_on

        beta = sts.norm.ppf(1 - self.alpha)
        self.beta = beta

        # TODO Update NSphere to handle n_dim case with order
        # Create sphere
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            _circle = np.stack((_x, _y), axis=1)
            sphere_points = beta * _circle

        else:
            sphere = NSphere(dim=n_dim, n_samples=n_points)
            sphere_points = beta * sphere.unit_sphere_points

        # Get probabilities for coordinates
        norm_cdf = sts.norm.cdf(sphere_points)

        # Inverse procedure. Get coordinates from probabilities.
        p = norm_cdf
        coordinates = np.empty_like(p)

        coordinates[:, 0] = distributions[0].icdf(p[:, 0])

        for i in range(1, n_dim):
            if conditional_on[i] is None:
                coordinates[:, i] = distributions[i].icdf(p[:, i])
            else:
                cond_idx = conditional_on[i]
                coordinates[:, i] = distributions[i].icdf(p[:, i], given=coordinates[:, cond_idx])

        self.sphere_points = sphere_points
        self.coordinates = coordinates


class ISORMContour(Contour):   
    """
    Contour based on the inverse second-order reliability method.
       
    This method was proposed by Chai and Leira (2018) [1]_
       
    Parameters
    ----------
    model : MultivariateModel
        The model to be used to calculate the contour.
    alpha : float
        The exceedance probability. The probability that an observation falls 
        outside the environmental contour.
    n_points : int, optional
        Number of points on the contour. Defaults to 180.
    
    Attributes
    ----------
    coordinates : 
        Coordinates of the calculated contour.
    beta :
        Reliability index.  
    sphere_points : 
          Points of the sphere in U space [1]_ . 
    
    References
    ----------
    .. [1] Chai, W.; Leira, B.J. (2018)
        Environmental contours based on inverse SORM. Marine Structures Volume 60,
        pp. 34-51. DOI: 10.1016/j.marstruc.2018.03.007 .
       
    """
    
    def __init__(self, model, alpha, n_points=180):
        self.model = model
        self.alpha = alpha
        self.n_points = n_points
        super().__init__()

    def _compute(self, ):
        """
        Calculates coordinates using ISORM.

        """

        n_dim = self.model.n_dim
        n_points = self.n_points

        distributions = self.model.distributions
        conditional_on = self.model.conditional_on

        # Use the ICDF of a chi-squared distribution with n dimensions. For
        # reference see equation 20 in Chai and Leira (2018).
        beta = np.sqrt(sts.chi2.ppf(1 - self.alpha, n_dim))

        # Create sphere.
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            _circle = np.stack((_x, _y)).T
            sphere_points = beta * _circle

        else:
            sphere = NSphere(dim=n_dim, n_samples=n_points)
            sphere_points = beta * sphere.unit_sphere_points

        # Get probabilities for coordinates of shape.
        norm_cdf_per_dimension = [sts.norm.cdf(sphere_points[:, dim])
                                  for dim in range(n_dim)]

        # Inverse procedure. Get coordinates from probabilities.
        data = np.zeros((n_points, n_dim))

        for i in range(n_dim):
            dist = distributions[i]
            cond_idx = conditional_on[i]
            if cond_idx is None:
                data[:, i] = dist.icdf(norm_cdf_per_dimension[i])
            else:
                conditioning_values = data[:, cond_idx]
                for j in range(n_points):
                    data[j, i] = dist.icdf(norm_cdf_per_dimension[i][j],
                                           given=conditioning_values[j])

        self.beta = beta
        self.sphere_points = sphere_points
        self.coordinates = data


class HighestDensityContour(Contour):
    """
    Contour based on the highest density method.

    This method was proposed by Haselsteiner et. al (2017) [1]_

    Parameters
    ----------
    model : MultivariateModel
        The model to be used to calculate the contour.
    alpha : float
        The exceedance probability. The probability that an observation 
        falls outside the environmental contour.
    limits : list of tuples, optional
       The limits of the grid to use for calculation. One 2-element tuple 
       for each dimension of the model, containing the minimum and maximum 
       for that dimension. (min, max). If not given, reasonable values are 
       choosen using the models marginal_icdf as upper limit and 0 as lower
       limit.
    deltas : float or list of float, optional
       The step size of the grid to use for calculation. If a single float 
       is supplied it is used for all dimensions. If a list is supplied 
       there has to be one entry for each dimension of the model. Defaults 
       to 0.25% of the range defined by limits.
       
    Attributes
    ----------
    coordinates : ndarray
        Coordinates of the calculated contour.
        Shape: (number of points, number of dimensions).  
    cell_center_coordinates : list of array
        Points at which the grid is evaluated.
        A list with one entry for each dimension, each entry is an array with
        the cell centers for that dimension.
    fm : float
        Minimum probability density of the enclosed region / constant 
        probability density along the contour. 
                
    
    References
    ----------
    .. [1] Haselsteiner, A.F.; Ohlendorf, J.H.; Wosniok, W.; Thoben, K.D. (2017)
        Deriving environmental contours from highest density regions,
        Coastal Engineering, Volume 123. DOI: 10.1016/j.coastaleng.2017.03.002.
        
    """

    def __init__(self, model, alpha, limits=None, deltas=None):
    

        self.model = model
        self.alpha = alpha
        self.limits = limits
        self.deltas = deltas
        self._check_grid()
        super().__init__()

    def _check_grid(self):
        n_dim = self.model.n_dim
        limits = self.limits
        deltas = self.deltas

        if limits is None:
            alpha = self.alpha
            marginal_icdf = self.model.marginal_icdf
            non_exceedance_p = 1 - 0.2 ** n_dim * alpha
            limits = [(0, marginal_icdf(non_exceedance_p, dim))
                      for dim in range(n_dim)]
            # TODO use distributions lower bound instead of zero
        else:
            # Check limits length.
            if len(limits) != n_dim:
                raise ValueError("limits has to be of length equal to number of dimensions, "
                                 f"but len(limits)={len(limits)}, n_dim={n_dim}.")
        self.limits = limits

        if deltas is None:
            deltas = np.empty(shape=n_dim)
            # Set default cell size to 0.25 percent of the variable space.
            # This is losely based on the results from Fig. 7 in 10.1016/j.coastaleng.2017.03.002
            # In the considered variable space length of 20 a cell length of
            # 0.05 was sufficient --> 20 / 0.05 = 400. 1/400 = 0.0025
            relative_cell_size = 0.0025

            for i in range(n_dim):
                deltas[i] = (limits[i][1] - limits[i][0]) * relative_cell_size
        else:
            try:  # Check if deltas is an iterable
                iter(deltas)
                if len(deltas) != n_dim:
                    raise ValueError("deltas has do be either scalar, "
                                     "or list of length equal to number of dimensions, "
                                     f"but was list of length {len(deltas)}")
                deltas = list(deltas)
            except TypeError:  # asserts that deltas is scalar
                deltas = [deltas] * n_dim

        self.deltas = deltas

    def _compute(self):
        """
        Calculates coordinates using HDC.

        """
        
        limits = self.limits
        deltas = self.deltas
        n_dim = self.model.n_dim
        alpha = self.alpha

        # Create sampling coordinate arrays.
        cell_center_coordinates = []
        for i, lim_tuple in enumerate(limits):
            try:
                iter(lim_tuple)
                if len(lim_tuple) != 2:
                    raise ValueError("tuples in limits have to be of length 2 ( = (min, max)), "
                                     f"but tuple with index = {i}, has length = {len(lim_tuple)}.")

            except TypeError:
                raise ValueError("tuples in limits have to be of length 2 ( = (min, max)), "
                                 f"but tuple with index = {i}, has length = 1.")

            min_ = min(lim_tuple)
            max_ = max(lim_tuple)
            delta = deltas[i]
            samples = np.arange(min_, max_ + delta, delta)
            cell_center_coordinates.append(samples)

        f = self.cell_averaged_joint_pdf(cell_center_coordinates)  # TODO

        if np.isnan(f).any():
            raise ValueError("Encountered nan in cell averaged probabilty joint pdf. "
                             "Possibly invalid distribution parameters?")

        # Calculate probability per cell.
        cell_prob = f
        for delta in deltas:
            cell_prob *= delta

        # Calculate highest density region.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                HDR, prob_m = self.cumsum_biggest_until(cell_prob, 1 - alpha)
        except RuntimeWarning:
            HDR = np.ones_like(cell_prob)
            prob_m = 0
            warnings.warn("A probability of 1-alpha could not be reached. "
                          "Consider enlarging the area defined by limits or "
                          "setting n_years to a smaller value.",
                          RuntimeWarning, stacklevel=4)

        # Calculate fm from probability per cell.
        fm = prob_m
        for delta in deltas:
            fm /= delta

        structure = np.ones(tuple([3] * n_dim), dtype=bool)
        HDC = HDR - ndi.binary_erosion(HDR, structure=structure)

        labeled_array, n_modes = ndi.label(HDC, structure=structure)

        coordinates = []
        # Iterate over all partial contours and start at 1.
        for i in range(1, n_modes + 1):
            # Array of arrays with same length, one per dimension
            # containing the indice of the contour.
            partial_contour_indice = np.nonzero(labeled_array == i)

            # Calculate the values corresponding to the indice
            partial_coordinates = []
            for dimension, indice in enumerate(partial_contour_indice):
                partial_coordinates.append(cell_center_coordinates[dimension][indice])

            coordinates.append(partial_coordinates)

        is_single_contour = False
        if len(coordinates) == 1:
            is_single_contour = True
            coordinates = coordinates[0]

        self.cell_center_coordinates = cell_center_coordinates
        self.fm = fm

        if is_single_contour:
            if n_dim == 2:
                self.coordinates = np.array(
                    sort_points_to_form_continuous_line(*coordinates,
                                                        search_for_optimal_start=True)).T
            else:
                self.coordinates = np.array(coordinates).T
        else:
            self.coordinates = coordinates
            # TODO raise warning

    @staticmethod
    def cumsum_biggest_until(array, limit):        
        """
        Find biggest elements to sum to reach limit.

        Sorts array, and calculates the cumulative sum.
        Returns a boolean array with the same shape as array indicating the
        fields summed to reach limit, as well as the last value added.

        Parameters
        ----------
        array : ndarray
            Array of arbitrary shape with all values >= 0.
        limit : float
            Value to sum up to.

        Returns
        -------
        summed_fields : ndarray, dtype=Bool
            Boolean array of shape like array with True if element was used in summation.
        last_summed : float
            Element that was added last to the sum.

        Raises
        ------
        ValueError
            If `array` contains nan.
        Notes
        ------
        A ``RuntimeWarning`` is raised if the limit cannot be reached by summing all values.
        
        """

        flat_array = np.ravel(array)
        if np.isnan(flat_array).any():
            raise ValueError("array contains nan.")

        sort_inds = np.argsort(flat_array, kind="mergesort")[::-1]
        sort_vals = flat_array[sort_inds]

        cum_sum = np.cumsum(sort_vals)

        if cum_sum[-1] < limit:
            warnings.warn("The limit could not be reached.", RuntimeWarning, stacklevel=1)

        summed_flat_inds = sort_inds[cum_sum <= limit]

        summed_fields = np.zeros(array.shape)

        summed_fields[np.unravel_index(summed_flat_inds, shape=array.shape)] = 1

        last_summed = array[np.unravel_index(summed_flat_inds[-1], shape=array.shape)]

        return summed_fields, last_summed

    def cell_averaged_joint_pdf(self, coords):  
        """
        Calculates the cell averaged joint probabilty density function.

        Multiplies the cell averaged probability densities of all distributions.
        
        Parameters
        ----------
        coords : list of array
            List with one coordinate array for each dimension.
            
        Returns
        -------
        fbar : array
            Joint cell averaged probability density function evaluated at coords.
            Cell averaged probability density function evaluated at coords.
            n dimensional array, where n is the number of dimensions of the 
            model used for calculation.
           

        """
        
        n_dim = len(coords)
        fbar = np.ones(((1,) * n_dim), dtype=np.float64)
        for dist_idx in range(n_dim):
            fbar = np.multiply(fbar, self.cell_averaged_pdf(dist_idx, coords))

        return fbar

    def cell_averaged_pdf(self, dist_idx, coords): 
        """
        Calculates the cell averaged probabilty density function of a single 
        distribution.

        Calculates the pdf by approximating it with the finite differential 
        quotient of the cumulative distributions function, evaluated at the 
        grid cells borders.
        i.e. :math:`f(x) \\approx \\frac{F(x+ 0.5\\Delta x) - F(x- 0.5\\Delta x) }{\\Delta x}`
        
        Parameters
        ----------
        dist_idx : int
            The index of the distribution to calcululate the pdf for.
        
        coords : list of array
            List with one coordinate array for each dimension.
           
        Returns
        -------
        fbar : array
            Cell averaged probability density function evaluated at coords.
            n dimensional array, where n is the number of dimensions of the 
            model used for calculation. All dimensions but, the dist_idx and 
            the cond_idx dimensions are of length 1. The dist_idx and cond_idx 
            dimensions are of length equal to the length of coords.


        """
        
        n_dim = len(coords)
        dist = self.model.distributions[dist_idx]
        cond_idx = self.model.conditional_on[dist_idx]

        dx = coords[dist_idx][1] - coords[dist_idx][0]

        cdf = dist.cdf
        fbar_out_shape = np.ones(n_dim, dtype=int)

        if cond_idx is None:  # independent variable
            # Calculate averaged pdf.
            lower = cdf(coords[dist_idx] - 0.5 * dx)
            upper = cdf(coords[dist_idx] + 0.5 * dx)
            fbar = (upper - lower)

            fbar_out_shape[dist_idx] = len(coords[dist_idx])

        else:
            dist_values = coords[dist_idx]
            cond_values = coords[cond_idx]
            fbar = np.empty((len(cond_values), len(dist_values)))
            for i, cond_value in enumerate(cond_values):
                lower = cdf(coords[dist_idx] - 0.5 * dx, given=cond_value)
                upper = cdf(coords[dist_idx] + 0.5 * dx, given=cond_value)
                fbar[i, :] = (upper - lower)

            fbar_out_shape[dist_idx] = len(coords[dist_idx])
            fbar_out_shape[cond_idx] = len(coords[cond_idx])

        fbar_out = fbar.reshape(fbar_out_shape)
        return fbar_out / dx

class DirectSamplingContour(Contour):
    """
    Direct sampling contour as introduced by Huseby et al. (2013) [1]_
    The provided direct sampling contour method only works for 2D models.

    Parameters
    ----------
    model : MultivariateModel
        The model to be used to calculate the contour.
    alpha : float
        The exceedance probability. The probability that an observation 
        falls outside the environmental contour.
    n : int, optional
        Number of data points that shall be Monte Carlo simulated.
    deg_step : float, optional
        Directional step in degrees. Defaults to 5.
    sample : 2-dimensional ndarray, optional
        Monte Carlo simulated environmental states. Array is of shape (n, d)
        with d being the number of variables and n being the number of
        observations.
    
    Attributes
    ----------
    coordinates : ndarray
        Coordinates of the calculated contour.
        Shape: (number of points, number of dimensions).          
    
    References
    ----------
    .. [1] Huseby, A.B.; Vanem, E.; Natvig, B. (2013)
        A new approach to environmental contours for ocean engineering applications
        based on direct Monte Carlo simulations,
        Ocean Engineering, Volume 60. DOI: doi.org/10.1016/j.oceaneng.2012.12.034

    """

    def __init__(self, model, alpha, n=100000, deg_step=5, sample=None):
        self.model = model
        self.alpha = alpha
        self.n = n
        self.deg_step = deg_step
        self.sample = sample
        super().__init__()

    def _compute(self):
        sample = self.sample
        n = self.n
        deg_step = self.deg_step
        alpha = self.alpha

        if self.model.n_dim != 2:
            raise NotImplementedError("DirectSamplingContour is currently only "
                                      "implemented for two dimensions.")

        if sample is None:
            sample = self.model.draw_sample(n)
            self.sample = sample
        x, y = sample.T

        # Calculate non-exceedance probability.
        # alpha = 1 - (1 / (self.return_period * 365.25 * 24 / self.state_duration))
        non_exceedance_p = 1 - alpha

        # Define the angles such the coordinates[0] and coordinates[1] will
        # be based on the exceedance plane with angle 0 deg, with 0 deg being
        # along the x-axis. Angles will increase counterclockwise in a xy-plot.
        # Not enirely sure why the + 2*rad_step is required, but tests show it.
        rad_step = deg_step * np.pi / 180
        angles = np.arange(0.5 * np.pi + 2 * rad_step, -1.5 * np.pi + rad_step,
                           -1 * rad_step)

        length_t = len(angles)
        r = np.zeros(length_t)

        # Find radius for each angle.
        i = 0
        while i < length_t:
            z = x * np.cos(angles[i]) + y * np.sin(angles[i])
            r[i] = np.quantile(z, non_exceedance_p)
            i = i + 1

        # Find intersection of lines.
        a = np.array(np.concatenate((angles, [angles[0]]), axis=0))
        r = np.array(np.concatenate((r, [r[0]]), axis=0))

        denominator = np.sin(a[2:]) * np.cos(a[1:len(a) - 1]) - \
                      np.sin(a[1:len(a) - 1]) * np.cos(a[2:])

        x_cont = (np.sin(a[2:]) * r[1:len(r) - 1]
                  - np.sin(a[1:len(a) - 1]) * r[2:]) / denominator
        y_cont = (-np.cos(a[2:]) * r[1:len(r) - 1]
                  + np.cos(a[1:len(a) - 1]) * r[2:]) / denominator

        self.coordinates = np.array([x_cont, y_cont]).T
