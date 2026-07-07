import json
import numpy as np
from tabulate import tabulate
np.set_printoptions(precision=15)

class Relaxation:
    """This class contains functions that calculates position of nodes, control points, li, xi, eta, phi, psi, P_matrix, A_matrix, gamma_vector, cartesian_velocity, C_p, C_L, C_mle, C_mc/4"""

    def __init__(self, json_file):
        self.json_file = json_file
        self.load_json()

    def load_json(self):
        """This function pulls in all the input values from the json"""
        with open(self.json_file, 'r') as json_handle:
            input_vals = json.load(json_handle)
            self.airfoils = input_vals['airfoils']
            self.alpha_deg = input_vals["alpha[deg]"]
            self.alpha_rad = np.radians(self.alpha_deg)
            self.V_inf = input_vals["freestream_velocity"]
            self.V_inf_vec = np.array([self.V_inf*np.cos(self.alpha_rad), self.V_inf*np.sin(self.alpha_rad)], dtype=float)
            self.initial_vortex = input_vals["initial_vortex"]
            self.acceptable_vort_error = input_vals["acceptable_vort_error"]
            self.acceptable_source_error = input_vals["acceptable_source_error"]
            self.acceptable_Kutta_error = input_vals["acceptable_Kutta_error"]
            self.max_relaxation_iterations = input_vals["max_relaxation_iterations"]
            self.vortex_relaxation_factor = input_vals["vortex_relaxation_factor"]
            self.source_relaxation_factor = input_vals["source_relaxation_factor"]

    def pull_nodes(self, i):
        """This function grabs the text files inside the json and turns them into arrays"""
        foil = self.airfoils[i]
        with open(foil, 'r') as text_handle:
            points = [list(map(float, line.strip().split())) for line in text_handle]
        self.control_points = np.array(points)
        self.N = len(self.control_points)

    def Argos_init(self):
        """Given the nodes and freestream velocity, compute unit tangent and unit normal vectors, Delta s, and K, and initialize sigma and omega for each control point"""
        ### Begin with initializing unit_tangent_matrix, unit_normal_matrix, Delta_s_vector, source_vector, and vortex_vector ###
        ### Initialize numpy arrays for each of the matrices and vectors
        self.unit_tangent_matrix = np.zeros((self.N,2))
        self.unit_normal_matrix = np.zeros((self.N,2))
        self.Delta_s_vector = np.zeros(self.N)
        self.source_vector = np.zeros(self.N)
        self.vortex_vector = np.zeros(self.N)
        ### loop through all control points ###
        for i in range(self.N):
            if (i==0):
                tangent_vector = self.control_points[i+1]-self.control_points[i]
                self.Delta_s_vector[i] = np.linalg.norm(tangent_vector)
            elif (i==self.N-1):
                tangent_vector = self.control_points[i]-self.control_points[i-1]
                self.Delta_s_vector[i] = np.linalg.norm(tangent_vector)
            else:
                tangent_vector = self.control_points[i+1]-self.control_points[i]
                p1 = self.control_points[i+1]-self.control_points[i]
                p2 = self.control_points[i]-self.control_points[i-1]
                self.Delta_s_vector[i] = 0.5*(np.linalg.norm(p1) + np.linalg.norm(p2))
            self.unit_tangent_matrix[i] = tangent_vector/np.linalg.norm(tangent_vector)
            normal_vector = np.array([-tangent_vector[1],tangent_vector[0]])
            self.unit_normal_matrix[i] = normal_vector/np.linalg.norm(normal_vector)
            self.source_vector[i] = -np.dot(self.unit_normal_matrix[i],self.V_inf_vec)
            self.vortex_vector[i] = self.initial_vortex
        # now initialize the Kernel "K" matrix, which is NxN. Each entry is a 2-D vector.
        print("initial source vector:\n", self.source_vector)
        print("initial vortex vector:\n", self.vortex_vector)
        self.init_Kernel()

    def init_Kernel(self):
        """creates the kernel matrix"""
        self.Kernel = np.zeros((self.N,self.N, 2)) # NxN matrix where each entry is a 2D vector.
        for i in range(self.N):
            P = self.control_points[i]
            for j in range(self.N):
                Q = self.control_points[j]
                r_QP = P-Q
                if i != j:
                    self.Kernel[i][j] = r_QP/(2*np.pi*np.dot(r_QP,r_QP))
                else:
                    self.Kernel[i][j] = r_QP
        print("kernel:\n", self.Kernel)
        return self.Kernel    

    def calc_v_matrix(self):
        """"""  
        for i in range(self.N):
            self.V_matrix[i] = self.V_inf_vec.copy()
            for j in range(self.N):
                Delta = self.Delta_s_vector[j]
                Kernel_first = self.Kernel[i][j][0]
                # print("Kernel First: ", Kernel_first)
                Kernel_second = self.Kernel[i][j][1]
                # print("Kernel Second: ", Kernel_second)
                source = self.source_vector[j]
                # print("Source in v mat: ", source)
                vortex = self.vortex_vector[j]
                # print("Vortex in v mat: ", vortex)
                if i==j:
                    self.V_matrix[i] += 0.5*source*self.unit_normal_matrix[j]
                else:
                    self.V_matrix[i][0] += (source*Kernel_first + vortex*Kernel_second)*Delta ### using first component of cross product
                    self.V_matrix[i][1] += (source*Kernel_second - vortex*Kernel_first)*Delta
        return self.V_matrix

    def Argos_saver_scheme(self):
        """"""
        ell = 0 # initialize count
        max_vortical_norm = 100 # initialize vortex residual
        max_source_norm = 100 # initialize source residual
        self.Res_vortex_vector = np.zeros((self.N))
        self.Res_source_vector = np.zeros((self.N))
        self.V_matrix = np.zeros((self.N,2)) # initialize velocity vector
        while ((max_vortical_norm > self.acceptable_vort_error) or (max_source_norm > self.acceptable_source_error)) and (ell < self.max_relaxation_iterations):
            self.calc_v_matrix()
            for i in range(self.N):
                ## 1. Evaluate total velocity components explicitly on the EXTERNAL face skin
                Vn_ext = np.dot(self.V_matrix[i], self.unit_normal_matrix[i])# + 0.5 * self.source_vector[i]
                Vs_ext = np.dot(self.V_matrix[i], self.unit_tangent_matrix[i])# + 0.5 * self.vortex_vector[i]
                ## 2. Compute Hunt's normal boundary velocity error (Target - Computed)
                epsilon_n = 0.0 - Vn_ext # 0.0 to remember that Hunt said epsilon_n = Vn-VnBC
                ## 3. Calculate Hunt's literal relaxation targets
                sigma_target = -np.dot(self.unit_normal_matrix[i], self.V_inf_vec) + epsilon_n # exactly as hunt defines it
                omega_target = Vs_ext - np.dot(self.unit_tangent_matrix[i], self.V_inf_vec) # exactly as hunt defines it
                ## 4. Residuals: current value minus Hunt's target destination
                self.Res_source_vector[i] = self.source_vector[i] - sigma_target
                self.Res_vortex_vector[i] = self.vortex_vector[i] - omega_target
            # i = 0
            for i in range(self.N):
                self.vortex_vector[i] -= (self.vortex_relaxation_factor*self.Res_vortex_vector[i])
                self.source_vector[i] -= (self.source_relaxation_factor*self.Res_source_vector[i])
            if abs(self.vortex_vector[self.N-1]+self.vortex_vector[0]) > self.acceptable_Kutta_error:
                print("vort vec[N-1]: ", self.vortex_vector[self.N-1])
                print("vort vec[0]: ", self.vortex_vector[0])
                for i in range(self.N):
                    self.vortex_vector[i] -= 0.5*(self.vortex_vector[self.N-1]+self.vortex_vector[0])
            max_vortical_norm = np.linalg.norm(self.Res_vortex_vector, ord=np.inf)
            max_source_norm = np.linalg.norm(self.Res_source_vector,ord=np.inf)
            ell += 1

    def program(self, i):
        """This is a run function that uses all of the functions above."""
        self.pull_nodes(i)
        self.Argos_init()
        self.Argos_saver_scheme()

if __name__ == "__main__":
    NACA_object = Relaxation("airfoils.json")
    NACA_object.program(0)