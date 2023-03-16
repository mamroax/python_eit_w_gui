import numpy as np
import warnings
from numpy import sqrt
from dataclasses import dataclass
from typing import Union
from itertools import combinations
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, linalg, coo_matrix
import scipy.linalg as la
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from typing import Callable, Union, List, Tuple
from abc import ABC, abstractmethod


class PyEITMesh:
    """
    Pyeit buid-in mesh object

    Parameters
    ----------
    node : np.ndarray
        node of the mesh of shape (n_nodes, 2), (n_nodes, 3)
    element : np.ndarray
        elements of the mesh of shape (n_elem, 3) for 2D mesh, (n_elem, 4) for 3D mesh
    perm : Union[int, float, np.ndarray], optional
        permittivity on elements; shape (n_elems,), by default `None`.
        If `None`, a uniform permittivity on elements with a value 1 will be generated.
        If perm is int or float, uniform permittivity on elements with value of perm will be generated.
    el_pos : np.ndarray
        node corresponding to each electrodes of shape (n_el, 1)
    ref_node : int
        reference node. ref_node should not be on electrodes, default 0.
    """
    def __new__(cls, *args, **kwargs):
        print("Hello from __new__")
        return super().__new__(cls)

    def __init__(self, element, node, perm, el_pos, ref_node):
        print("Hello from __init__")
        self.element = element
        self.node = node
        self.perm = perm
        self.el_pos = el_pos
        self.ref_node = ref_node

    node: np.ndarray
    element: np.ndarray
    perm: Union[int, float, np.ndarray] = None
    el_pos: np.ndarray = np.arange(16)
    ref_node: int = 0

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.element = self._check_element(self.element)
        self.node = self._check_node(self.node)
        self.perm = self.get_valid_perm(self.perm)
        self.ref_node = self._check_ref_node(self.ref_node)

    def print_stats(self):
        """
        Print mesh or tetrahedral status

        Parameters
        ----------
        p: array_like
            coordinates of nodes (x, y) in 2D, (x, y, z) in 3D
        t: array_like
            connectives forming elements

        Notes
        -----
        a simple function for illustration purpose only.
        print the status (size) of nodes and elements
        """
        text_2D_3D = "3D" if self.is_3D else "2D"
        print(f"{text_2D_3D} mesh status:")
        print(f"{self.n_nodes} nodes, {self.n_elems} elements")

    def _check_element(self, element: np.ndarray) -> np.ndarray:
        """
        Check nodes element
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_elem,3)
            nodes [x,y,z] ; shape (n_nodes,4)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """
        if not isinstance(element, np.ndarray):
            raise TypeError(f"Wrong type of {element=}, expected an ndarray")
        if element.ndim != 2:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray with 2 dimensions"
            )
        if element.shape[1] not in [3, 4]:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )

        return element

    def _check_node(self, node: np.ndarray) -> np.ndarray:
        """
        Check nodes shape
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_nodes,2) (in that case z will be set 0)
            nodes [x,y,z] ; shape (n_nodes,3)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """
        if not isinstance(node, np.ndarray):
            raise TypeError(f"Wrong type of {node=}, expected an ndarray")
        if node.ndim != 2:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray with 2 dimensions"
            )
        if node.shape[1] not in [2, 3]:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )
        # convert nodes [x,y] to nodes [x,y,0]
        if node.shape[1] == 2:
            node = np.hstack((node, np.zeros((node.shape[0], 1))))

        return node

    def get_valid_perm(self, perm: Union[int, float, np.ndarray] = None) -> np.ndarray:
        """
        Return a valid permittivity on element

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            Permittivity on elements ; shape (n_elems,), by default `None`.
            If `None`, a uniform permittivity on elements with a value 1 will be used.
            If perm is int or float, uniform permittivity on elements will be used.

        Returns
        -------
        np.ndarray
            permittivity on elements ; shape (n_elems,)

        Raises
        ------
        TypeError
            check if perm passed as ndarray has a shape (n_elems,)
        """

        if perm is None:
            return np.ones(self.n_elems, dtype=float)
        elif isinstance(perm, (int, float)):
            return np.ones(self.n_elems, dtype=float) * perm

        if not isinstance(perm, np.ndarray) or perm.shape != (self.n_elems,):
            raise TypeError(
                f"Wrong type/shape of {perm=}, expected an ndarray; shape ({self.n_elems}, )"
            )
        return perm

    def _check_ref_node(self, ref: int = 0) -> int:
        """
        Return a valid reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference node, by default 0
            If the choosen node is on electrode node, a node-list in
            np.arange(0, len(el_pos)+1) will be checked iteratively until
            a non-electrode node is selected.

        returns
        -------
        int
            valid reference electrode node
        """
        default_ref = np.setdiff1d(np.arange(len(self.el_pos) + 1), self.el_pos)[0]
        return ref if ref not in self.el_pos else default_ref
        # assert ref < self.n_nodes

    def set_ref_node(self, ref: int = 0) -> None:
        """
        Set reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference electrode
        """
        self.ref_node = self._check_ref_node(ref)

    @property
    def n_nodes(self) -> int:
        """
        Returns
        -------
        int
            number of nodes contained in the mesh
        """
        return self.node.shape[0]

    @property
    def n_elems(self) -> int:
        """
        Returns
        -------
        int
            number of elements contained in the mesh
        """
        return self.element.shape[0]

    @property
    def n_vertices(self) -> int:
        """
        Returns
        -------
        int
            number of vertices of the elements contained in the mesh
        """
        return self.element.shape[1]

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            number of electrodes
        """
        return self.el_pos.shape[0]

    @property
    def elem_centers(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            center of the nodes [x,y,z]; shape (n_elems,3)
        """
        return np.mean(self.node[self.element], axis=1)

    @property
    def is_3D(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            True if the mesh is a 3D mesh (use elements with 4 vertices)
        """
        return self.n_vertices == 4

    @property
    def is_2D(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            True if the mesh is a 2D mesh (use elements with 3 vertices)
        """
        return self.n_vertices == 3


@dataclass
class PyEITProtocol:
    """
    EIT Protocol buid-in protocol object

    Parameters
    ----------
    ex_mat: np.ndarray
        excitation matrix (pairwise)
    meas_mat: np.ndarray
        measurement matrix (differential pairs)
    keep_ba: np.ndarray
        boolean array index for keeping measurements
    """

    ex_mat: np.ndarray
    meas_mat: np.ndarray
    keep_ba: np.ndarray

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.ex_mat = self._check_ex_mat(self.ex_mat)
        self.meas_mat = self._check_meas_mat(self.meas_mat)
        self.keep_ba = self._check_keep_mat(self.keep_ba)

    def _check_ex_mat(self, ex_mat: np.ndarray) -> np.ndarray:
        """
        Check/init stimulation

        Parameters
        ----------
        ex_mat : np.ndarray
            stimulation/excitation matrix, of shape (n_exc, 2).
            If single stimulation (ex_line) is passed only a list of length 2
            and np.ndarray of size 2 will be treated.

        Returns
        -------
        np.ndarray
            stimulation matrix

        Raises
        ------
        TypeError
            Only accept, list of length 2, np.ndarray of size 2,
            or np.ndarray of shape (n_exc, 2)
        """
        if isinstance(ex_mat, list) and len(ex_mat) == 2:
            # case ex_line has been passed instead of ex_mat
            ex_mat = np.array([ex_mat]).reshape((1, 2))  # build a 2D array
        elif isinstance(ex_mat, np.ndarray) and ex_mat.size == 2:
            # case ex_line np.ndarray has been passed instead of ex_mat
            ex_mat = ex_mat.reshape((-1, 2))

        if not isinstance(ex_mat, np.ndarray):
            raise TypeError(f"Wrong type of {type(ex_mat)=}, expected an ndarray;")
        if ex_mat.ndim != 2 or ex_mat.shape[1] != 2:
            raise TypeError(f"Wrong shape of {ex_mat.shape=}, should be (n_exc, 2);")

        return ex_mat

    def _check_meas_mat(self, meas_mat: np.ndarray) -> np.ndarray:
        """
        Check measurement pattern

        Parameters
        ----------
        n_exc : int
            number of excitations/stimulations
        meas_pattern : np.ndarray, optional
           measurements pattern / subtract_row pairs [N, M] to check; shape (n_exc, n_meas_per_exc, 2)

        Returns
        -------
        np.ndarray
            measurements pattern / subtract_row pairs [N, M]; shape (n_exc, n_meas_per_exc, 2)

        Raises
        ------
        TypeError
            raised if meas_pattern is not a np.ndarray of shape (n_exc, : , 2)
        """
        if not isinstance(meas_mat, np.ndarray):
            raise TypeError(f"Wrong type of {type(meas_mat)=}, expected an ndarray;")
        # test shape is something like (n_exc, :, 2)
        if meas_mat.ndim != 3 or meas_mat.shape[::2] != (self.n_exc, 2):
            raise TypeError(
                f"Wrong shape of {meas_mat.shape=}, should be ({self.n_exc}, n_meas_per_exc, 2);"
            )

        return meas_mat

    def _check_keep_mat(self, keep_ba: np.ndarray) -> np.ndarray:
        """check keep boolean array"""
        if not isinstance(keep_ba, np.ndarray):
            raise TypeError(f"Wrong type of {type(keep_ba)=}, expected an ndarray;")

        return keep_ba

    @property
    def n_exc(self) -> int:
        """
        Returns
        -------
        int
            number of excitation
        """
        return self.ex_mat.shape[0]

    @property
    def n_meas(self) -> int:
        """
        Returns
        -------
        int
            number of measurements per excitations
        """
        return self.meas_mat.shape[1]

    @property
    def n_meas_tot(self) -> int:
        """
        Returns
        -------
        int
            total amount of measurements
        """
        return self.n_meas * self.n_exc

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            number of electrodes used in the excitation and
        """
        return max(max(self.ex_mat.flatten()), max(self.meas_mat.flatten())) + 1


class SolverNotReadyError(BaseException):
    """Is raised if solver.setup() not called before using solver"""


class EitBase(ABC):
    """
    Base EIT solver.
    """

    def __init__(
        self,
        mesh: PyEITMesh,
        protocol: PyEITProtocol,
    ) -> None:
        """
        An EIT solver.

        WARNING: Before using it run solver.setup() to set the solver ready!

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object
        protocol: PyEITProtocol
            measurement object
        """
        # build forward solver
        self.fwd = EITForward(mesh=mesh, protocol=protocol)

        # initialize other parameters
        self.params = None
        self.xg = None
        self.yg = None
        self.mask = None
        # user must run solver.setup() manually to get correct H
        self.H = None
        self.is_ready = False

    @property
    def mesh(self) -> PyEITMesh:
        return self.fwd.mesh

    # # if needed protocol attributes can be accessed by using self.protocol
    # # instead of self.fwd.protocol
    # @property
    # def protocol(self)->PyEITProtocol:
    #     return self.fwd.protocol

    @abstractmethod
    def setup(self) -> None:
        """
        Setup EIT solver

        1. memory parameters in self.params
        2. compute some other stuff needed for 3.
        3. compute self.H used for solving inv problem by using
            >> self.H=self._compute_h()
        4. set flag self.is_ready to `True`
        """

    @abstractmethod
    def _compute_h(self) -> np.ndarray:
        """
        Compute H matrix for solving inv problem

        To be used in self.setup()
        >> self.H=self._compute_h()

        Returns
        -------
        np.ndarray
            H matrix
        """

    def solve(
        self,
        v1: np.ndarray,
        v0: np.ndarray,
        normalize: bool = False,
        log_scale: bool = False,
    ) -> np.ndarray:
        """
        Dynamic imaging (conductivities imaging)

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame, d = H(v1 - v0)
        normalize: Bool, optional
            true for conducting normalization, by default False
        log_scale: Bool, optional
            remap reconstructions in log scale, by default False

        Raises
        ------
            SolverNotReadyError: raised if solver not ready
            (see self._check_solver_is_ready())

        Returns
        -------
        ds: np.ndarray
            complex-valued np.ndarray, changes of conductivities
        """
        self._check_solver_is_ready()
        dv = self._normalize(v1, v0) if normalize else v1 - v0
        ds = -np.dot(self.H, dv.transpose())  # s = -Hv
        if log_scale:
            ds = np.exp(ds) - 1.0
        return ds

    def map(self, dv: np.ndarray) -> np.ndarray:
        """
        (NOT USED, Deprecated?) simple mat using projection matrix

        return -H*dv, dv should be normalized.

        Parameters
        ----------
        dv : np.ndarray
            voltage measurement frame difference (reference frame - current frame)

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            -H*dv
        """
        self._check_solver_is_ready()
        return -np.dot(self.H, dv.transpose())

    def _check_solver_is_ready(self) -> None:
        """
        Check if solver is ready for solving

        Addtionaly test also if self.H not `None`

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready
        """
        if not self.is_ready or self.H is None:
            msg = "User must first run {type(self).__name__}.setup() before imaging."
            raise SolverNotReadyError(msg)

    def _normalize(self, v1: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """
        Normalize current frame using the amplitude of the reference frame.
        Boundary measurements v are complex-valued, we can use the real part of v,
        np.real(v), or the absolute values of v, np.abs(v).

        Parameters
        ----------
        v1 : np.ndarray
            current frame, can be a Nx192 matrix where N is the number of frames
        v0 : np.ndarray
            referenced frame, which is a row vector

        Returns
        -------
        np.ndarray
            Normalized current frame difference dv
        """
        return (v1 - v0) / np.abs(v0)

class JAC(EitBase):
    """A sensitivity-based EIT imaging class"""

    def setup(
        self,
        p: float = 0.20,
        lamb: float = 0.001,
        method: str = "kotre",
        perm: Union[int, float, np.ndarray] = None,
        jac_normalized: bool = False,
    ) -> None:
        """
        Setup JAC solver

        Jacobian matrix based reconstruction.

        Parameters
        ----------
        p : float, optional
            JAC parameters, by default 0.20
        lamb : float, optional
            JAC parameters, by default 0.001
        method : str, optional
            regularization methods ("kotre", "lm", "dgn" ), by default "kotre"
        perm : Union[int, float, np.ndarray], optional
            If perm is not None, a prior of perm distribution is used to build jac
        jac_normalized : bool, optional
            normalize the jacobian using f0 computed from input perm, by
            default False
        """
        # passing imaging parameters
        self.params = {
            "p": p,
            "lamb": lamb,
            "method": method,
            "jac_normalize": jac_normalized,
        }
        # pre-compute H0 for dynamical imaging
        # H = (J.T*J + R)^(-1) * J.T
        self.J, self.v0 = self.fwd.compute_jac(perm=perm, normalize=jac_normalized)
        self.H = self._compute_h(self.J, p, lamb, method)
        self.is_ready = True

    def _compute_h(
        self, jac: np.ndarray, p: float, lamb: float, method: str = "kotre"
    ) -> np.ndarray:
        """
        Compute self.H matrix for JAC solver

        JAC method of dynamic EIT solver:
            H = (J.T*J + lamb*R)^(-1) * J.T

        Parameters
        ----------
        jac : np.ndarray
            Jacobian
        p : float
            Regularization parameter, the p in R=diag(diag(JtJ) ** p)
        lamb : float
            Regularization parameter, the lambda in (JtJ + lambda*R)^{-1}
        method : str, optional
            Regularization method, ("kotre", "lm", "dgn" ), by default "kotre".
            Note that the name method="kotre" uses regularization alike the one
            in adler-dai-lionheart-2007 (pp4):
            "Temporal Image Reconstruction in Electrical Impedance Tomography",
            it regularize the diagonal of JtJ by an exponential parameter p.

        Returns
        -------
        np.ndarray
            H matrix, pseudo-inverse matrix of JAC
        """
        j_w_j = np.dot(jac.transpose(), jac)
        if method == "kotre":
            # p=0   : noise distribute on the boundary ('dgn')
            # p=0.5 : noise distribute on the middle
            # p=1   : noise distribute on the center ('lm')
            r_mat = np.diag(np.diag(j_w_j) ** p)
        elif method == "lm":
            # Marquardt–Levenberg, 'lm' for short
            # or can be called NOSER, DLS
            r_mat = np.diag(np.diag(j_w_j))
        else:
            # Damped Gauss Newton, 'dgn' for short
            r_mat = np.eye(jac.shape[1])

        # build H
        return np.dot(la.inv(j_w_j + lamb * r_mat), jac.transpose())

    def solve_gs(self, v1: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """
        Solving by weighted frequency

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            complex-valued np.ndarray, changes of conductivities
        """
        self._check_solver_is_ready()
        a = np.dot(v1, v0) / np.dot(v0, v0)
        dv = v1 - a * v0
        # return ds average epsilon on element
        return -np.dot(self.H, dv.transpose())

    def jt_solve(
        self, v1: np.ndarray, v0: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """
        a 'naive' back projection using the transpose of Jac.
        This scheme is the one published by kotre (1989), see note [1].

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame
        normalize : bool, optional
            flag to log-normalize the current frame difference dv, by default
            True. The input (dv) and output (ds) is log-normalized.

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            complex-valued np.ndarray, changes of conductivities

        Notes
        -----
            [1] Kotre, C. J. (1989).
                A sensitivity coefficient method for the reconstruction of
                electrical impedance tomograms.
                Clinical Physics and Physiological Measurement,
                10(3), 275--281. doi:10.1088/0143-0815/10/3/008

        """
        self._check_solver_is_ready()
        if normalize:
            dv = np.log(np.abs(v1) / np.abs(v0)) * np.sign(v0.real)
        else:
            dv = (v1 - v0) * np.sign(v0.real)
        # s_r = J^Tv_r
        ds = -np.dot(self.J.conj().T, dv)
        return np.exp(ds) - 1.0

    def gn(
        self,
        v: np.ndarray,
        x0: Union[int, float, np.ndarray] = None,
        maxiter: int = 1,
        gtol: float = 1e-4,
        p: float = None,
        lamb: float = None,
        lamb_decay: float = 1.0,
        lamb_min: float = 0.0,
        method: str = "kotre",
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Gaussian Newton Static Solver
        You can use a different p, lamb other than the default ones in setup

        Parameters
        ----------
        v : np.ndarray
            boundary measurement
        x0 : Union[int, float, np.ndarray], optional
            initial permittivity guess, by default None
            (see Foward._get_perm for more details, in fem.py)
        maxiter : int, optional
            number of maximum iterations, by default 1
        gtol : float, optional
            convergence threshold, by default 1e-4
        p : float, optional
            JAC parameters (can be overridden), by default None
        lamb : float, optional
            JAC parameters (can be overridden), by default None
        lamb_decay : float, optional
            decay of lamb0, i.e., lamb0 = lamb0 * lamb_delay of each iteration,
            by default 1.0
        lamb_min : float, optional
            minimal value of lamb, by default 0.0
        method : str, optional
            regularization methods ("kotre", "lm", "dgn" ), by default "kotre"
        verbose : bool, optional
            verbose flag, by default False

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            Complex-valued conductivities, sigma

        Note
        ----
        Gauss-Newton Iterative solver,
            x1 = x0 - (J^TJ + lamb*R)^(-1) * r0
        where:
            R = diag(J^TJ)**p
            r0 (residual) = real_measure - forward_v
        """
        self._check_solver_is_ready()
        if x0 is None:
            x0 = self.mesh.perm
        if p is None:
            p = self.params["p"]
        if lamb is None:
            lamb = self.params["lamb"]
        if method is None:
            method = self.params["method"]

        # convergence test
        x0_norm = np.linalg.norm(x0)

        for i in range(maxiter):

            # forward solver,
            jac, v0 = self.fwd.compute_jac(x0)
            # Residual
            r0 = v - v0

            # Damped Gaussian-Newton
            h_mat = self._compute_h(jac, p, lamb, method)

            # update
            d_k = np.dot(h_mat, r0)
            x0 = x0 - d_k

            # convergence test
            c = np.linalg.norm(d_k) / x0_norm
            if c < gtol:
                break

            if verbose:
                print("iter = %d, lamb = %f, gtol = %f" % (i, lamb, c))

            # update regularization parameter
            # lambda can be given in user defined decreasing lists
            lamb *= lamb_decay
            lamb = max(lamb, lamb_min)
        return x0

    def project(self, ds: np.ndarray) -> np.ndarray:
        """
        Project ds using spatial difference filter (deprecated)

        Parameters
        ----------
        ds : np.ndarray
            delta sigma (conductivities)

        Returns
        -------
        np.ndarray
            _description_
        """
        """project ds using spatial difference filter (deprecated)

        Parameters
        ----------
        ds: np.ndarray
            delta sigma (conductivities)

        Returns
        -------
        np.ndarray
        """
        d_mat = sar(self.mesh.element)
        return np.dot(d_mat, ds)




class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh: PyEITMesh) -> None:
        """
        FEM forward solver.
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes.

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object

        Note
        ----
        The nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        """
        self.mesh = mesh
        # coefficient matrix [initialize]
        self.se = calculate_ke(self.mesh.node, self.mesh.element)
        self.assemble_pde(self.mesh.perm)

    def assemble_pde(self, perm: Union[int, float, np.ndarray]) -> None:
        """
        assemble PDE

        Parameters
        ----------
        perm : Union[int, float, np.ndarray]
            permittivity on elements ; shape (n_tri,).
            if `None`, assemble_pde is aborded

        """
        if perm is None:
            return
        perm = self.mesh.get_valid_perm(perm)
        self.kg = assemble(
            self.se, self.mesh.element, perm, self.mesh.n_nodes, ref=self.mesh.ref_node
        )

    def solve(self, ex_line: np.ndarray = None) -> np.ndarray:
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for a
        excitation contained specified by `ex_line` (Neumann BC)

        Parameters
        ----------
        ex_line : np.ndarray, optional
            stimulation/excitation matrix, of shape (2,)

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_pts,)

        Notes
        -----
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.
        """
        # using natural boundary conditions
        b = np.zeros(self.mesh.n_nodes)
        b[self.mesh.el_pos[ex_line]] = [1, -1]

        # solve
        return linalg.spsolve(self.kg, b)




class EITForward(Forward):
    """EIT Forward simulation, depends on mesh and protocol"""

    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol) -> None:
        """
        EIT Forward Solver

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object
        protocol: PyEITProtocol
            measurement object

        Notes
        -----
        The Jacobian and the boundary voltages used the SIGN information,
        for example, V56 = V6 - V5 = -V65. If you are using absolute boundary
        voltages for imaging, you MUST normalize it with the signs of v0
        under each current-injecting pattern.
        """
        self._check_mesh_protocol_compatibility(mesh, protocol)

        # FEM solver
        super().__init__(mesh=mesh)

        # EIT measurement protocol
        self.protocol = protocol

    def _check_mesh_protocol_compatibility(
        self, mesh: PyEITMesh, protocol: PyEITProtocol
    ) -> None:
        """
        Check if mesh and protocol are compatible

        - #1 n_el in mesh >=  n_el in protocol
        - #2 .., TODO if necessary

        Raises
        ------
        ValueError
            if protocol is not compatible to the mesh
        """
        # n_el in mesh should be >=  n_el in protocol
        m_n_el = mesh.n_el
        p_n_el = protocol.n_el

        if m_n_el != p_n_el:
            warnings.warn(
                f"The mesh use {m_n_el} electrodes, and the protocol use only {p_n_el} electrodes",
                stacklevel=2,
            )

        if m_n_el < p_n_el:
            raise ValueError(
                f"Protocol is not compatible with mesh :\
The mesh use {m_n_el} electrodes, and the protocol use only {p_n_el} electrodes "
            )

    def solve_eit(
        self,
        perm: Union[int, float, np.ndarray] = None,
    ) -> np.ndarray:
        """
        EIT simulation, generate forward v measurements

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of forward v measurements will be
            based on the permittivity of the mesh, self.mesh.perm
        Returns
        -------
        v: np.ndarray
            simulated boundary voltage measurements; shape(n_exe*n_el,)
        """
        self.assemble_pde(perm)
        v = np.zeros(
            (self.protocol.n_exc, self.protocol.n_meas), dtype=self.mesh.perm.dtype
        )
        for i, ex_line in enumerate(self.protocol.ex_mat):
            f = self.solve(ex_line)
            v[i] = subtract_row(f[self.mesh.el_pos], self.protocol.meas_mat[i])

        return v.reshape(-1)

    def compute_jac(
        self,
        perm: Union[int, float, np.ndarray] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobian matrix and initial boundary voltage meas.
        extimation v0

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of Jacobian matrix will be based
            on the permittivity of the mesh, self.mesh.perm
        normalize : bool, optional
            flag for Jacobian normalization, by default False.
            If True the Jacobian is normalized

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Jacobian matrix, initial boundary voltage meas. extimation v0

        """
        # update k if necessary and calculate r=inv(k)
        self.assemble_pde(perm)
        r_el = la.inv(self.kg.toarray())[self.mesh.el_pos]

        # calculate v, jac per excitation pattern (ex_line)
        _jac = np.zeros(
            (self.protocol.n_exc, self.protocol.n_meas, self.mesh.n_elems),
            dtype=self.mesh.perm.dtype,
        )
        v = np.zeros(
            (self.protocol.n_exc, self.protocol.n_meas), dtype=self.mesh.perm.dtype
        )
        for i, ex_line in enumerate(self.protocol.ex_mat):
            f = self.solve(ex_line)
            v[i] = subtract_row(f[self.mesh.el_pos], self.protocol.meas_mat[i])
            ri = subtract_row(r_el, self.protocol.meas_mat[i])
            # Build Jacobian matrix column wise (element wise)
            #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
            for (e, ijk) in enumerate(self.mesh.element):
                _jac[i, :, e] = np.dot(np.dot(ri[:, ijk], self.se[e]), f[ijk])

        # measurement protocol
        jac = np.vstack(_jac)
        v0 = v.reshape(-1)

        # Jacobian normalization: divide each row of J (J[i]) by abs(v0[i])
        if normalize:
            jac = jac / np.abs(v0[:, None])
        return jac, v0

    def compute_b_matrix(
        self,
        perm: Union[int, float, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute back-projection mappings (smear matrix)

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of smear matrix will be based
            on the permittivity of the mesh, self.mesh.perm

        Returns
        -------
        np.ndarray
            back-projection mappings (smear matrix); shape(n_exc, n_pts, 1), dtype= bool
        """
        self.assemble_pde(perm)
        b_mat = np.zeros((self.protocol.n_exc, self.protocol.n_meas, self.mesh.n_nodes))

        for i, ex_line in enumerate(self.protocol.ex_mat):
            f = self.solve(ex_line=ex_line)
            f_el = f[self.mesh.el_pos]
            # build bp projection matrix
            # 1. we can either smear at the center of elements, using
            #    >> fe = np.mean(f[:, self.tri], axis=1)
            # 2. or, simply smear at the nodes using f
            b_mat[i] = _smear(f, f_el, self.protocol.meas_mat[i])

        return np.vstack(b_mat)


class DISTMESH:
    """class for distmesh"""

    def __init__(
        self,
        fd,
        fh,
        h0=0.1,
        p_fix=None,
        bbox=None,
        density_ctrl_freq=30,
        deltat=0.1,
        dptol=0.001,
        ttol=0.1,
        Fscale=1.2,
        verbose=False,
    ):
        """initial distmesh class

        Parameters
        ----------
        fd : str
            function handle for distance of boundary
        fh : str
            function handle for distance distributions
        h0 : float, optional
            Distance between points in the initial distribution p0,
            default=0.1 For uniform meshes, h(x,y) = constant,
            the element size in the final mesh will usually be
            a little larger than this input.
        p_fix : array_like, optional
            fixed points, default=[]
        bbox : array_like, optional
            bounding box for region, bbox=[xmin, ymin, xmax, ymax].
            default=[-1, -1, 1, 1]
        density_ctrl_freq : int, optional
            cycles of iterations of density control, default=20
        deltat : float, optional
            mapping forces to distances, default=0.2
        dptol : float, optional
            exit criterion for minimal distance all points moved, default=0.01
        ttol : float, optional
            enter criterion for re-delaunay the lattices, default=0.1
        Fscale : float, optional
            rescaled string forces, default=1.2
            if set too small, points near boundary will be pushed back
            if set too large, points will be pushed towards boundary

        Notes
        -----
        """
        # shape description
        self.fd = fd
        self.fh = fh
        self.h0 = h0

        # a small gap, allow points who are slightly outside of the region
        self.deps = np.sqrt(np.finfo(np.double).eps) * h0
        self.geps = 1e-1 * h0

        # control the distmesh computation flow
        self.densityctrlfreq = density_ctrl_freq
        self.dptol = dptol
        self.ttol = ttol
        self.Fscale = Fscale
        self.deltat = deltat

        # default bbox is 2D
        if bbox is None:
            bbox = [[-1, -1], [1, 1]]
        # p : coordinates (x,y) or (x,y,z) of meshes
        self.n_dim = np.shape(bbox)[1]
        p = bbox2d_init(h0, bbox)


        # control debug messages
        self.verbose = verbose
        self.num_triangulate = 0
        self.num_density = 0
        self.num_move = 0

        # keep points inside (minus distance) with a small gap (geps)
        p = p[fd(p) < self.geps]  # pylint: disable=E1136

        # rejection points by sampling on fh
        r0 = 1.0 / fh(p) ** self.n_dim
        selection = np.random.rand(p.shape[0]) < (r0 / np.max(r0))
        p = p[selection]

        # specify fixed points
        if p_fix is None:
            p_fix = []
        self.pfix = p_fix
        self.nfix = len(p_fix)

        # remove duplicated points of p and p_fix
        # avoid overlapping of mesh points
        if self.nfix > 0:
            p = remove_duplicate_nodes(p, p_fix, self.geps)
            p = np.vstack([p_fix, p])

        # store p and N
        self.N = p.shape[0]
        self.p = p

        # initialize pold with inf: it will be re-triangulate at start
        self.pold = np.inf * np.ones((self.N, self.n_dim))

        # build edges list for triangle or tetrahedral. i.e., in 2D triangle
        # edge_combinations is [[0, 1], [1, 2], [2, 0]]
        self.edge_combinations = list(combinations(range(self.n_dim + 1), 2))

        # triangulate, generate simplices and bars
        self.triangulate()

    def is_retriangulate(self):
        """test whether re-triangulate is needed"""
        return np.max(dist(self.p - self.pold)) > (self.h0 * self.ttol)

    def triangulate(self):
        """retriangle by delaunay"""
        self.debug("enter triangulate = ", self.num_triangulate)
        self.num_triangulate += 1
        # pnew[:] = pold[:] makes a new copy, not reference
        self.pold = self.p.copy()

        # triangles where the points are arranged counterclockwise
        # QJ parameter so tuples don't exceed boundary
        tri = Delaunay(self.p, qhull_options="QJ").simplices
        pmid = np.mean(self.p[tri], axis=1)

        # keeps only interior points
        t = tri[self.fd(pmid) < -self.geps]
        # extract edges (bars)
        bars = t[:, self.edge_combinations].reshape((-1, 2))
        # sort and remove duplicated edges, eg (1,2) and (2,1)
        # note : for all edges, non-duplicated edge is boundary edge
        bars = np.sort(bars, axis=1)
        bars_tuple = bars.view([("", bars.dtype)] * bars.shape[1])

        self.bars = np.unique(bars_tuple).view(bars.dtype).reshape((-1, 2))
        self.t = t

    def bar_length(self):
        """the forces of bars (python is by-default row-wise operation)"""
        # two node of a bar
        bars_a, bars_b = self.p[self.bars[:, 0]], self.p[self.bars[:, 1]]
        barvec = bars_a - bars_b

        # L : length of bars, must be column ndarray (2D)
        L = dist(barvec).reshape((-1, 1))
        # density control on bars
        hbars = self.fh((bars_a + bars_b) / 2.0).reshape((-1, 1))
        # L0 : desired lengths (Fscale matters!)
        L0 = hbars * self.Fscale * sqrt(np.sum(L**2) / np.sum(hbars**2))

        return L, L0, barvec

    def bar_force(self, L, L0, barvec):
        """forces on bars"""
        # abs(forces)
        F = np.maximum(L0 - L, 0)
        # normalized and vectorized forces
        Fvec = F * (barvec / L)
        # now, we get forces and sum them up on nodes
        # using sparse matrix to perform automatic summation
        # rows : left, left, right, right (2D)
        # : left, left, left, right, right, right (3D)
        # cols : x, y, x, y (2D)
        # : x, y, z, x, y, z (3D)
        data = np.hstack([Fvec, -Fvec])
        if self.n_dim == 2:
            rows = self.bars[:, [0, 0, 1, 1]]
            cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 0, 1]]))
        else:
            rows = self.bars[:, [0, 0, 0, 1, 1, 1]]
            cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 2, 0, 1, 2]]))
        # sum nodes at duplicated locations using sparse matrices
        Ftot = csr_matrix(
            (data.reshape(-1), [rows.reshape(-1), cols.reshape(-1)]),
            shape=(self.N, self.n_dim),
        )
        Ftot = Ftot.toarray()
        # zero out forces at fixed points, as they do not move
        Ftot[0 : len(self.pfix)] = 0

        return Ftot

    def density_control(self, L, L0, dscale=3.0):
        """
        Density control - remove points that are too close
        L0 : Kx1, L : Kx1, bars : Kx2
        bars[L0 > 2*L] only returns bar[:, 0] where L0 > 2L
        """
        self.debug("enter density control = ", self.num_density)
        self.num_density += 1
        # print(self.num_density, self.p.shape)
        # quality control
        ixout = (L0 > dscale * L).ravel()
        ixdel = np.setdiff1d(self.bars[ixout, :].reshape(-1), np.arange(self.nfix))
        self.p = self.p[np.setdiff1d(np.arange(self.N), ixdel)]
        # Nold = N
        self.N = self.p.shape[0]
        self.pold = np.inf * np.ones((self.N, self.n_dim))
        # print('density control ratio : %f' % (float(N)/Nold))

    def move_p(self, Ftot):
        """update p"""
        self.debug("  number of moves = ", self.num_move)
        self.num_move += 1
        # move p along forces
        self.p += self.deltat * Ftot

        # if there is any point ends up outside
        # move it back to the closest point on the boundary
        # using the numerical gradient of distance function
        d = self.fd(self.p)
        ix = d > 0
        if ix.any():
            self.p[ix] = edge_project(self.p[ix], self.fd, self.geps)

        # check whether convergence : no big movements
        delta_move = self.deltat * np.max(dist(Ftot[d < -self.geps]))
        self.debug("  delta_move = ", delta_move)
        score = delta_move < self.dptol * self.h0

        return score

    def debug(self, *args):
        """print debug messages"""
        if self.verbose:
            print(*args)

def tet_volume(xyz):
    """calculate the volume of tetrahedron"""
    s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]
    v_tot = (1.0 / 6.0) * la.det(s)
    return v_tot

def edge_grad(p, fd, h0=1.0):
    """
    project points back on the boundary (where fd=0) using numerical gradient
    3D, ND compatible

    Parameters
    ----------
    p : array_like
        points on 2D, 3D
    fd : str
        function handler of distances
    h0 : float
        minimal distance

    Returns
    -------
    array_like
        gradients of points on the boundary

    Note
    ----
        numerical gradient:
        f'_x = (f(p+delta_x) - f(p)) / delta_x
        f'_y = (f(p+delta_y) - f(p)) / delta_y
        f'_z = (f(p+delta_z) - f(p)) / delta_z

        you should specify h0 according to your actual mesh size
    """
    # d_eps = np.sqrt(np.finfo(float).eps)*h0
    # d_eps = np.sqrt(np.finfo(float).eps)
    d_eps = 1e-8 * h0

    # get dimensions
    if np.ndim(p) == 1:
        p = p[:, np.newaxis]

    # distance
    d = fd(p)

    # calculate the gradient of each axis
    ndim = p.shape[1]
    pts_xyz = np.repeat(p, ndim, axis=0)
    delta_xyz = np.repeat([np.eye(ndim)], p.shape[0], axis=0).reshape(-1, ndim)
    deps_xyz = d_eps * delta_xyz
    g_xyz = (fd(pts_xyz + deps_xyz) - np.repeat(d, ndim, axis=0)) / d_eps

    # normalize gradient, avoid divide by zero
    g = g_xyz.reshape(-1, ndim)
    g2 = np.sum(g**2, axis=1)

    # move unit
    g_num = g / g2[:, np.newaxis] * d[:, np.newaxis]

    return g_num

def edge_project(pts, fd, h0=1.0):
    """project points back on edge"""
    g_vec = edge_grad(pts, fd, h0)
    return pts - g_vec

def remove_duplicate_nodes(p, pfix, geps):
    """remove duplicate points in p who are closed to pfix. 3D, ND compatible

    Parameters
    ----------
    p : array_like
        points in 2D, 3D, ND
    pfix : array_like
        points that are fixed (can not be moved in distmesh)
    geps : float, optional (default=0.01*h0)
        minimal distance that two points are assumed to be identical

    Returns
    -------
    array_like
        non-duplicated points
    """
    for row in pfix:
        pdist = dist(p - row)
        # extract non-duplicated row slices
        p = p[pdist > geps]
    return p

def bbox2d_init(h0, bbox):
    """
    generate points in 2D bbox (not including the ending point of bbox)

    Parameters
    ----------
    h0 : float
        minimal distance of points
    bbox : array_like
        [[x0, y0],
         [x1, y1]]

    Returns
    -------
    array_like
        points in bbox
    """
    x, y = np.meshgrid(
        np.arange(bbox[0][0], bbox[1][0], h0),
        np.arange(bbox[0][1], bbox[1][1], h0 * sqrt(3) / 2.0),
        indexing="xy",
    )
    # shift even rows of x
    # shift every second row h0/2 to the right, therefore,
    # all points will be a distance h0 from their closest neighbors
    x[1::2, :] += h0 / 2.0
    # p : Nx2 ndarray
    p = np.array([x.ravel(), y.ravel()]).T
    return p


def breath(file_path, num_of_frames):
    try:
        breath_file = open(file_path)
        breath_matrix = [line.split("	") for line in breath_file]
        breath_points = []
        x = [i for i in range(0, 450, num_of_frames)]
        min_value_index = 0
        max_value_index = 0
        min_value = 1
        max_value = 0
        for i in range(0, len(breath_matrix), num_of_frames):
            summa = 0
            for k in range(num_of_frames):
                for j in breath_matrix[i+k]:
                    summa += float(j)
            average = summa/208/num_of_frames
            if average < min_value:
                min_value = average
                min_value_index = i
            if average > max_value:
                max_value = average
                max_value_index = i
            breath_points.append(average)
    except Exception:
        print('Error')
    finally:
        breath_file.close()
    return [x, breath_points, min_value_index, max_value_index]

def area_uniform(p):
    """uniform mesh distribution

    Parameters
    ----------
    p : array_like
        points coordinates

    Returns
    -------
    array_like
        ones

    """
    return np.ones(p.shape[0])

def thorax(pts):
    """
    thorax polygon signed distance function

    Thorax contour points coordinates are taken from
    a thorax simulation based on EIDORS
    """
    poly = [
        (0.0487, 0.6543),
        (0.1564, 0.6571),
        (0.2636, 0.6697),
        (0.3714, 0.6755),
        (0.479, 0.6686),
        (0.5814, 0.6353),
        (0.6757, 0.5831),
        (0.7582, 0.5137),
        (0.8298, 0.433),
        (0.8894, 0.3431),
        (0.9347, 0.2452),
        (0.9698, 0.1431),
        (0.9938, 0.0379),
        (1.0028, -0.0696),
        (0.9914, -0.1767),
        (0.9637, -0.281),
        (0.9156, -0.3771),
        (0.8359, -0.449),
        (0.7402, -0.499),
        (0.6432, -0.5463),
        (0.5419, -0.5833),
        (0.4371, -0.6094),
        (0.3308, -0.6279),
        (0.2243, -0.6456),
        (0.1168, -0.6508),
        (0.0096, -0.6387),
        (-0.098, -0.6463),
        (-0.2058, -0.6433),
        (-0.313, -0.6312),
        (-0.4181, -0.6074),
        (-0.5164, -0.5629),
        (-0.6166, -0.5232),
        (-0.7207, -0.4946),
        (-0.813, -0.4398),
        (-0.8869, -0.3614),
        (-0.933, -0.2647),
        (-0.9451, -0.1576),
        (-0.9425, -0.0498),
        (-0.9147, 0.0543),
        (-0.8863, 0.1585),
        (-0.8517, 0.2606),
        (-0.8022, 0.3565),
        (-0.7413, 0.4455),
        (-0.6664, 0.5231),
        (-0.5791, 0.5864),
        (-0.4838, 0.6369),
        (-0.3804, 0.667),
        (-0.2732, 0.6799),
        (-0.1653, 0.6819),
        (-0.0581, 0.6699),
    ]
    poly_obj = Polygon(poly)
    return fd_polygon(poly_obj, pts)

def fd_polygon(poly, pts):
    """return signed distance of polygon"""
    pts_ = [Point(p) for p in pts]
    # calculate signed distance
    dist = [poly.exterior.distance(p) for p in pts_]
    sign = np.sign([-int(poly.contains(p)) + 0.5 for p in pts_])

    return sign * dist

def circle(pts, pc=None, r=1.0):
    """
    Distance function for the circle centered at pc = [xc, yc]

    Parameters
    ----------
    pts : array_like
        points on 2D
    pc : array_like, optional
        center of points
    r : float, optional
        radius

    Returns
    -------
    array_like
        distance of (points - pc) - r

    Note
    ----
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    if pc is None:
        pc = [0, 0]
    return dist(pts - pc) - r

def dist(p):
    """distances to origin of nodes. '3D', 'ND' compatible
        distances of points to origin
    """
    if p.ndim == 1:
        d = np.sqrt(np.sum(p**2))
    else:
        d = np.sqrt(np.sum(p**2, axis=1))
    return d

def create_mesh(
    n_el: int = 16,
    fd: Callable = None,
    fh: Callable = area_uniform,
    h0: float = 0.1,
    p_fix: np.ndarray = None,
    bbox: np.ndarray = None,
) -> PyEITMesh:
    """
    Generating 2D/3D meshes using distmesh (pyEIT built-in)

    Parameters
    ----------
    n_el: int
        number of electrodes (point-type electrode)
    fd: function
        distance function (circle in 2D, ball in 3D)
    fh: function
        mesh size quality control function
    p_fix: NDArray
        fixed points
    bbox: NDArray
        bounding box
    h0: float
        initial mesh size, default=0.1

    Returns
    -------
    PyEITMesh
        mesh object
    """

    # test conditions if fd or/and bbox are none
    bbox = np.array([[-1, -1], [1, 1]]) # here i may have some problems////////////////////////////////////////////
    # list is converted to Numpy array so we can use it then (calling shape method..)
    bbox = np.array(bbox)
    n_dim = bbox.shape[1]  # bring dimension
    fd = circle

    if n_dim not in [2, 3]:
        raise TypeError("distmesh only supports 2D or 3D")
    if bbox.shape[0] != 2:
        raise TypeError("please specify lower and upper bound of bbox")

    p_fix = np.array(
        [
            (-0.098, -0.6463),
            (-0.4181, -0.6074),
            (-0.7207, -0.4946),
            (-0.933, -0.2647),
            (-0.9147, 0.0543),
            (-0.8022, 0.3565),
            (-0.5791, 0.5864),
            (-0.1653, 0.6819),
            (0.1564, 0.6571),
            (0.5814, 0.6353),
            (0.8298, 0.433),
            (0.9698, 0.1431),
            (0.9914, -0.1767),
            (0.8359, -0.449),
            (0.5419, -0.5833),
            (0.2243, -0.6456),
        ]
    )

    # 1. build mesh
    p, t = build(fd, fh, pfix=p_fix, bbox=bbox, h0=h0)
    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)
    # 3. generate electrodes, the same as p_fix (top n_el)
    el_pos = np.arange(n_el)
    return PyEITMesh(element=t, node=p, perm=None, el_pos=el_pos, ref_node=0)


def build(
    fd, fh, pfix=None, bbox=None, h0=0.1, densityctrlfreq=10, maxiter=500, verbose=False
):
    """main function for distmesh

    See Also
    --------
    DISTMESH : main class for distmesh

    Parameters
    ----------
    fd, fh : function
        signed distance and quality control function
    pfix : ndarray
        fixed points on the boundary, usually used for electrodes positions
    maxiter : int, optional
        maximum iteration numbers, default=1000

    Returns
    -------
    p : array_like
        points on 2D bbox
    t : array_like
        triangles describe the mesh structure

    Notes
    -----
    there are many python or hybrid python + C implementations in github,
    this implementation is merely implemented from scratch
    using PER-OLOF PERSSON's Ph.D thesis and SIAM paper.

    .. [1] P.-O. Persson, G. Strang, "A Simple Mesh Generator in MATLAB".
       SIAM Review, Volume 46 (2), pp. 329-345, June 2004

    Also, the user should be aware that, equal-edged tetrahedron cannot fill
    space without gaps. So, in 3D, you can lower dptol, or limit the maximum
    iteration steps.
    """
    # parsing arguments
    # make sure : g_Fscale < 1.5
    mode_3D = False
    if bbox is None:
        g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.2, 0.2
    else:
        # perform error check on bbox
        bbox = np.array(bbox)
        if (bbox.ndim == 1) or (bbox.shape[1] not in [2, 3]):
            raise TypeError("only 2D, 3D are supported, bbox = ", bbox)
        if bbox.shape[0] != 2:
            raise TypeError("please specify lower and upper bound of bbox")
        # default parameters for 2D
        g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.3, 0.2

    # initialize distmesh
    dm = DISTMESH(
        fd,
        fh,
        h0=h0,
        p_fix=pfix,
        bbox=bbox,
        density_ctrl_freq=densityctrlfreq,
        deltat=g_deltat,
        dptol=g_dptol,
        ttol=g_ttol,
        Fscale=g_Fscale,
        verbose=verbose,
    )

    # now iterate to push to equilibrium
    for i in range(maxiter):
        if dm.is_retriangulate():
            # print("triangulate = %d" % dm.num_triangulate)
            dm.triangulate()

        # calculate bar forces
        L, L0, barvec = dm.bar_length()

        # density control
        if (i % densityctrlfreq) == 0 and (L0 > 2 * L).any():
            dm.density_control(L, L0)
            # continue to triangulate
            continue

        # calculate bar forces
        Ftot = dm.bar_force(L, L0, barvec)

        # update p
        converge = dm.move_p(Ftot)
        # the stopping ctriterion (movements interior are small)
        if converge:
            break

    # at the end of iteration, (p - pold) is small, so we recreate delaunay
    dm.triangulate()

    # you should remove duplicate nodes and triangles
    return dm.p, dm.t

def check_order(no2xy, el2no):
    """
    loop over all elements, calculate the Area of Elements (aoe)
    if AOE > 0, then the order of element is correct
    if AOE < 0, reorder the element

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity

    Returns
    -------
    NDArray
        ae, area of each element

    Notes
    -----
    tetrahedron should be parsed that the sign of volume is [1, -1, 1, -1]
    """
    el_num, n_vertices = np.shape(el2no)
    # select ae function
    if n_vertices == 3:
        _fn = tri_area
    elif n_vertices == 4:
        _fn = tet_volume
    # calculate ae and re-order tri if necessary
    for ei in range(el_num):
        no = el2no[ei, :]
        xy = no2xy[no, :]
        v = _fn(xy)
        # if CCW, area should be positive, otherwise,
        if v < 0:
            el2no[ei, [1, 2]] = el2no[ei, [2, 1]]

    return el2no

def tri_area(xy):
    """
    return area of a triangle, given its tri-coordinates xy

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    float
        area of this element
    """
    s = xy[[2, 0]] - xy[[1, 2]]
    a_tot = 0.50 * la.det(s)
    # (should be positive if tri-points are counter-clockwise)
    return a_tot

def create_protocol(
    n_el: int = 16,
    dist_exc: Union[int, List[int]] = 1,
    step_meas: int = 1,
    parser_meas: Union[str, List[str]] = "std",
) -> PyEITProtocol:
    """
    Return an EIT protocol, comprising an excitation and a measuremnet pattern

    Parameters
    ----------
    n_el : int, optional
        number of total electrodes, by default 16
    dist_exc : Union[int, List[int]], optional
        distance (number of electrodes) of A to B, by default 1
        For 'adjacent'- or 'neighbore'-mode (default) use `1` , and
        for 'apposition'-mode use `n_el/2`. (see `build_exc_pattern`)
        if a list of integer is passed the excitation will bee stacked together.
    step_meas : int, optional
    measurement method (two adjacent electrodes are used for measuring), by default 1 (adjacent).
        (see `build_meas_pattern`)
    parser_meas : Union[str, List[str]], optional
        parsing the format of each frame in measurement/file, by default 'std'.
        (see `build_meas_pattern`)

    Returns
    -------
    PyEITProtocol
        EIT protocol object

    Raises
    ------
    TypeError
        if dist_exc is not list or an int
    """
    if isinstance(dist_exc, int):
        dist_exc = [dist_exc]

    if not isinstance(dist_exc, list):
        raise TypeError(f"{type(dist_exc)=} should be a List[int]")

    _ex_mat = [build_exc_pattern_std(n_el, dist) for dist in dist_exc]
    ex_mat = np.vstack(_ex_mat)

    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)

def build_exc_pattern_std(n_el: int = 16, dist: int = 1) -> np.ndarray:
    """
    Generate scan matrix, `ex_mat` ( or excitation pattern), see notes

    Parameters
    ----------
    n_el : int, optional
        number of electrodes, by default 16
    dist : int, optional
        distance (number of electrodes) of A to B, by default 1
        For 'adjacent'- or 'neighbore'-mode (default) use `1` , and
        for 'apposition'-mode use `n_el/2` (see Examples).

    Returns
    -------
    np.ndarray
        stimulation matrix; shape (n_exc, 2)

    Notes
    -----
        - in the scan of EIT (or stimulation matrix), we use 4-electrodes
        mode, where A, B are used as positive and negative stimulation
        electrodes and M, N are used as voltage measurements.
        - `1` (A) for positive current injection, `-1` (B) for negative current
        sink

    Examples
    --------
        n_el=16
        if mode=='neighbore':
            ex_mat = build_exc_pattern(n_el=n_el)
        elif mode=='apposition':
            ex_mat = build_exc_pattern(dist=n_el/2)

    WARNING
    -------
        `ex_mat` is a local index, where it is ranged from 0...15, within the
        range of the number of electrodes. In FEM applications, you should
        convert `ex_mat` to global index using the (global) `el_pos` parameters.
    """
    return np.array([[i, np.mod(i + dist, n_el)] for i in range(n_el)])

def build_meas_pattern_std(
    ex_mat: np.ndarray,
    n_el: int = 16,
    step: int = 1,
    parser: Union[str, List[str]] = "std",
) -> np.ndarray:
    """
    Build the measurement pattern (subtract_row-voltage pairs [N, M])
    for all excitations on boundary electrodes.

    we direct operate on measurements or Jacobian on electrodes,
    so, we can use LOCAL index in this module, do not require el_pos.

    Notes
    -----
    ABMN Model.
    A: current driving electrode,
    B: current sink,
    M, N: boundary electrodes, where v_diff = v_n - v_m.

    Parameters
    ----------
    ex_mat : np.ndarray
        Nx2 array, [positive electrode, negative electrode]. ; shape (n_exc, 2)
    n_el : int, optional
        number of total electrodes, by default 16
    step : int, optional
        measurement method (two adjacent electrodes are used for measuring), by default 1 (adjacent)
    parser : Union[str, List[str]], optional
        parsing the format of each frame in measurement/file, by default 'std'
        if parser contains 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrode start index 'A'.
        if parser contains 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
        if parser contains 'meas_current', the measurements on current carrying
        electrodes are allowed. Otherwise the measurements on current carrying
        electrodes are discarded (like 'no_meas_current' option in EIDORS3D).

    Returns
    -------
    diff_op: np.ndarray
        measurements pattern / subtract_row pairs [N, M]; shape (n_exc, n_meas_per_exc, 2)
    keep_ba: np.ndarray
        (n_exc*n_meas_per_exc,) boolean array
    """
    if not isinstance(parser, list):  # transform parser into list
        parser = [parser]
    meas_current = "meas_current" in parser
    fmmu_rotate = any(p in ("fmmu", "rotate_meas") for p in parser)

    diff_op, keep_ba = [], []
    for ex_line in ex_mat:
        a, b = ex_line[0], ex_line[1]
        i0 = a if fmmu_rotate else 0
        m = (i0 + np.arange(n_el)) % n_el
        n = (m + step) % n_el
        meas_pattern = np.vstack([n, m]).T

        diff_keep = np.logical_and.reduce((m != a, m != b, n != a, n != b))
        keep_ba.append(diff_keep)
        if not meas_current:
            meas_pattern = meas_pattern[diff_keep]
        diff_op.append(meas_pattern)

    return np.array(diff_op), np.array(keep_ba).ravel()

def calculate_ke(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    Calculate local stiffness matrix on all elements.

    Parameters
    ----------
    pts: np.ndarray
        Nx2 (x,y) or Nx3 (x,y,z) coordinates of points
    tri: np.ndarray
        Mx3 (triangle) or Mx4 (tetrahedron) connectivity of elements

    Returns
    -------
    np.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    """
    n_tri, n_vertices = tri.shape

    # check dimension
    # '3' : triangles
    # '4' : tetrahedrons
    if n_vertices == 3:
        _k_local = _k_triangle
    elif n_vertices == 4:
        _k_local = _k_tetrahedron
    else:
        raise TypeError("The num of vertices of elements must be 3 or 4")

    # default data types for ke
    ke_array = np.zeros((n_tri, n_vertices, n_vertices))
    for ei in range(n_tri):
        no = tri[ei, :]
        xy = pts[no]

        # compute the KIJ (permittivity=1.)
        ke = _k_local(xy)
        ke_array[ei] = ke

    return ke_array

def _k_triangle(xy: np.ndarray) -> np.ndarray:
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: np.ndarray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    np.ndarray
        local stiffness matrix
    """
    # edges (vector) of triangles
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

    # area of triangles. Note, abs is removed since version 2020,
    # user must make sure all triangles are CCW (conter clock wised).
    # at = 0.5 * np.linalg.det(s[[0, 1]])
    at = 0.5 * det2x2(s[0], s[1])

    # Local stiffness matrix (e for element)
    return np.dot(s, s.T) / (4.0 * at)

def det2x2(s1: np.ndarray, s2: np.ndarray) -> float:
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]

def assemble(
    ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0
) -> np.ndarray:
    """
    Assemble the stiffness matrix (using sparse matrix)

    Parameters
    ----------
    ke: np.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    tri: np.ndarray
        the structure of mesh
    perm: np.ndarray
        n_tri x 1 conductivities on elements
    n_pts: int
        number of nodes
    ref: int, optional
        reference electrode, by default 0

    Returns
    -------
    np.ndarray
        NxN array of complex stiffness matrix

    Notes
    -----
    you may use sparse matrix (IJV) format to automatically add the local
    stiffness matrix to the global matrix.
    """
    n_tri, n_vertices = tri.shape

    # New: use IJV indexed sparse matrix to assemble K (fast, prefer)
    # index = np.array([np.meshgrid(no, no, indexing='ij') for no in tri])
    # note: meshgrid is slow, using handcraft sparse index, for example
    # let tri=[[1, 2, 3], [4, 5, 6]], then indexing='ij' is equivalent to
    # row = [1, 1, 1, 2, 2, 2, ...]
    # col = [1, 2, 3, 1, 2, 3, ...]
    row = np.repeat(tri, n_vertices).ravel()
    col = np.repeat(tri, n_vertices, axis=0).ravel()
    data = np.array([ke[i] * perm[i] for i in range(n_tri)]).ravel()

    # set reference nodes before constructing sparse matrix
    if 0 <= ref < n_pts:
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        # K[ref, :] = 0, K[:, ref] = 0
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        # K[ref, ref] = 1.0
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    # for efficient sparse inverse (csc)
    return csr_matrix((data, (row, col)), shape=(n_pts, n_pts))

def subtract_row(v: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """
    Build the voltage differences on axis=1 using the meas_pattern.
    v_diff[k] = v[i, :] - v[j, :]

    New implementation 33% less computation time

    Parameters
    ----------
    v: np.ndarray
        Nx1 boundary measurements vector or NxM matrix; shape (n_exc,n_el,1)
    meas_pattern: np.ndarray
        Nx2 subtract_row pairs; shape (n_exc, n_meas, 2)

    Returns
    -------
    np.ndarray
        difference measurements v_diff
    """
    return v[meas_pattern[:, 0]] - v[meas_pattern[:, 1]]

def _smear(f: np.ndarray, fb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Build smear matrix B for bp for one exitation

    used for the smear matrix computation by @ChabaneAmaury

    Parameters
    ----------
    f: np.ndarray
        potential on nodes
    fb: np.ndarray
        potential on adjacent electrodes
    pairs: np.ndarray
        electrodes numbering pairs

    Returns
    -------
    B: np.ndarray
        back-projection matrix
    """
    # Replacing the code below by a faster implementation in Numpy
    f_min = np.minimum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    f_max = np.maximum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    return (f_min < f) & (f <= f_max)

def _k_tetrahedron(xy: np.ndarray) -> np.ndarray:
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: np.ndarray
        (x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner, see notes.

    Returns
    -------
    np.ndarray
        local stiffness matrix

    Notes
    -----
    A tetrahedron is described using [0, 1, 2, 3] (local node index) or
    [171, 27, 9, 53] (global index). Counterclockwise (CCW) is defined
    such that the barycentric coordinate of face (1->2->3) is positive.
    """
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

    # volume of the tetrahedron, Note abs is removed since version 2020,
    # user must make sure all tetrahedrons are CCW (counter clock wised).
    vt = 1.0 / 6 * la.det(s[[0, 1, 2]])

    # calculate area (vector) of triangle faces
    # re-normalize using alternative (+,-) signs
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])

    # local (e for element) stiffness matrix
    return np.dot(a, a.transpose()) / (36.0 * vt)

def sar(el2no: np.ndarray) -> np.ndarray:
    """
    Extract spatial difference matrix on the neighbors of each element
    in 2D fem using triangular mesh.

    Parameters
    ----------
    el2no : np.ndarray
        triangle structures

    Returns
    -------
    np.ndarray
        SAR matrix
    """
    ne = el2no.shape[0]
    d_mat = np.eye(ne)
    for i in range(ne):
        ei = el2no[i, :]
        #
        i0 = np.argwhere(el2no == ei[0])[:, 0]
        i1 = np.argwhere(el2no == ei[1])[:, 0]
        i2 = np.argwhere(el2no == ei[2])[:, 0]
        idx = np.unique(np.hstack([i0, i1, i2]))
        # build row-i
        for j in idx:
            d_mat[i, j] = -1
        nn = idx.size - 1
        d_mat[i, i] = nn
    return d_mat

def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    """
    (2D/3D) compatible.

    Interp values on points using values on simplex,
    a simplex can be triangle or tetrahedron.
    The areas/volumes are used as weights.

    f_n = (sum_e r_e*S_e) / (sum_e S_e)

    where r_e is the value on triangles who share the node n,
    S_e is the area of triangle e.

    Parameters
    ----------
    pts_values: np.ndarray
        Nx1 array, real/complex valued
    sim: np.ndarray
        Mx3, Mx4 array, elements or simplex
        triangles denote connectivity [[i, j, k]]
        tetrahedrons denote connectivity [[i, j, m, n]]
    sim_value: np.ndarray

    Notes
    -----
    This function is similar to pdeprtni of MATLAB pde.
    """
    N = pts.shape[0]
    M, dim = sim.shape
    # calculate the weights
    # triangle/tetrahedron must be CCW (recommended), then a is positive
    if dim == 3:
        weight_func = tri_area
    elif dim == 4:
        weight_func = tet_volume
    weights = weight_func(pts, sim)
    # build tri->pts matrix, could be accelerated using sparse matrix
    row = np.ravel(sim)
    col = np.repeat(np.arange(M), dim)  # [0, 0, 0, 1, 1, 1, ...]
    data = np.repeat(weights, dim)
    e2n_map = coo_matrix((data, (row, col)), shape=(N, M)).tocsr()
    # map values from elements to nodes
    # and re-weight by the sum of the areas/volumes of adjacent elements
    f = e2n_map.dot(sim_values)
    w = np.sum(e2n_map.toarray(), axis=1)

    return f / w