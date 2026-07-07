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
            """Grabs text files with headers and handles both comma and space separators cleanly"""
            foil = self.airfoils[i]
            points = []
            with open(foil, 'r') as text_handle:
                for line in text_handle:
                    line_str = line.strip()
                    # Skip empty lines or the header row
                    if not line_str or 'x' in line_str.lower():
                        continue
                    # Replace commas with spaces to normalize the string
                    normalized_line = line_str.replace(',', ' ')
                    # Split by whitespace (handles spaces, tabs, and multiple spaces)
                    try:
                        coords = list(map(float, normalized_line.split()))
                        points.append(coords)
                    except ValueError:
                        # Safeguard for any other non-numeric lines
                        continue
            self.nodes = np.array(points)
            self.N = len(self.nodes)
            self.M = self.N - 1 # number of control points
            self.control_points = np.zeros((self.M, 2))
            for j in range(self.M):
                self.control_points[j] = 0.5 * (self.nodes[j] + self.nodes[j+1])
            x, y = self.nodes[:, 0], self.nodes[:, 1]
            diff_x = np.diff(x)
            diff_y = np.diff(y)
            self.L_vals = np.sqrt(diff_x**2 + diff_y**2)

    def Argos_init(self):
        """Compute unit tangent and unit normal vectors, Delta s, and initialize sigma and omega per panel"""
        self.unit_tangent_matrix = np.zeros((self.M, 2))
        self.unit_normal_matrix = np.zeros((self.M, 2))
        self.Delta_s_vector = np.zeros(self.M)
        self.source_vector = np.zeros(self.M)
        self.vortex_vector = np.zeros(self.M)
        # Loop through all M panels to compute geometry relative to panel vertices (nodes)
        for i in range(self.M):
            panel_vector = self.nodes[i+1] - self.nodes[i]
            self.Delta_s_vector[i] = np.linalg.norm(panel_vector)
            # Unit tangent
            self.unit_tangent_matrix[i] = panel_vector / self.Delta_s_vector[i]
            # Unit normal (outward facing assuming standard counter-clockwise/clockwise node order)
            self.unit_normal_matrix[i] = np.array([-self.unit_tangent_matrix[i][1], self.unit_tangent_matrix[i][0]])
            # Initial boundary condition matching
            self.source_vector[i] = -np.dot(self.unit_normal_matrix[i], self.V_inf_vec)
            self.vortex_vector[i] = self.initial_vortex
        # initialize xi_eta and phi_psi matrices
        self.calc_xi_eta_phi_psi_matrices()
        # initialize dx and dy 
        self.calc_dx_dy_vectors()

    def calc_xi_eta_phi_psi_matrices(self):
        """"""
        x_nodes, y_nodes = self.nodes[:, 0], self.nodes[:, 1]
        self.phi_matrix = np.zeros((self.M, self.M))
        self.psi_matrix = np.zeros((self.M, self.M))
        for i in range(self.M):
            control_x = self.control_points[i, 0]
            control_y = self.control_points[i, 1]
            for j in range(self.M):
                l_j = self.L_vals[j]
                xi, eta = self.calc_xi_eta(j, l_j, x_nodes, y_nodes, control_x, control_y)
                self.phi_matrix[i][j], self.psi_matrix[i][j] = self.calc_phi_psi(l_j, eta, xi)

    def calc_dx_dy_vectors(self):
        """"""
        x_nodes, y_nodes = self.nodes[:, 0], self.nodes[:, 1]
        self.dx_vector = np.zeros((self.M))
        self.dy_vector = np.zeros((self.M))
        for j in range(self.M):
            self.dx_vector[j] = x_nodes[j+1] - x_nodes[j]
            self.dy_vector[j] = y_nodes[j+1] - y_nodes[j]
            
    def calc_xi_eta(self, j, l_j, x_nodes, y_nodes,control_x, control_y):
        """"""
        xi = (1/l_j)*((x_nodes[j+1]-x_nodes[j])*(control_x-x_nodes[j]) + (y_nodes[j+1]-y_nodes[j])*(control_y-y_nodes[j]))
        eta = (1/l_j)*(-(y_nodes[j+1]-y_nodes[j])*(control_x-x_nodes[j])+(x_nodes[j+1]-x_nodes[j])*(control_y-y_nodes[j]))
        return xi, eta
    
    def calc_phi_psi(self, l_j, eta, xi):
        """"""
        phi = np.arctan2(eta * l_j, eta**2 + xi**2 - xi * l_j)
        psi = 0.5 * np.log((xi**2 + eta**2) / ((xi - l_j)**2 + eta**2))
        return phi, psi

    def calc_v_matrix(self):
        """Calculates total baseline velocity at each control point using analytic panel integrals"""
        x_nodes, y_nodes = self.nodes[:, 0], self.nodes[:, 1]
        for i in range(self.M):
            # 1. Start with the uniform freestream velocity
            self.V_matrix[i] = self.V_inf_vec.copy()
            for j in range(self.M):
                l_j = self.L_vals[j]
                # Singular/Self-influence check
                if i == j: # ignore self-influence
                    # Baseline flat self-induction at the geometric center is zero.
                    # The discontinuous half-jumps are added during residual calculation.
                    pass
                else:
                    # get the correct phi and psi pre-computed values
                    phi, psi = self.phi_matrix[i][j], self.psi_matrix[i][j]
                    # 3. Calculate local induced components for constant strength distributions
                    # Constant Source Panel induction in local coordinates:
                    u_source_local = (self.source_vector[j] * psi) / (2.0 * np.pi)
                    v_source_local = (self.source_vector[j] * phi) / (2.0 * np.pi)
                    # Constant Vortex Panel induction in local coordinates:
                    u_vortex_local = (self.vortex_vector[j] * phi) / (2.0 * np.pi)
                    v_vortex_local = -(self.vortex_vector[j] * psi) / (2.0 * np.pi)
                    # Total local velocity components
                    u_local = u_source_local + u_vortex_local
                    v_local = v_source_local + v_vortex_local
                    # 4. Rotate local velocities back to global coordinates 
                    dx = self.dx_vector[j]
                    dy = self.dy_vector[j]
                    V_x_induced = (dx*u_local - dy*v_local)/l_j
                    V_y_induced = (dy*u_local + dx*v_local)/l_j
                    self.V_matrix[i][0] += V_x_induced
                    self.V_matrix[i][1] += V_y_induced
        return self.V_matrix

    def Argos_saver_scheme(self):
        """Executes Hunt's SAVER dual relaxation algorithm with explicit external face jump corrections"""
        ell = 0 
        max_vortical_norm = 100 
        max_source_norm = 100 
        self.Res_vortex_vector = np.zeros(self.M)
        self.Res_source_vector = np.zeros(self.M)
        self.V_matrix = np.zeros((self.M, 2)) 
        while ((max_vortical_norm > self.acceptable_vort_error) or (max_source_norm > self.acceptable_source_error)) and (ell < self.max_relaxation_iterations):
            ## Calculate baseline integrated induction field
            self.calc_v_matrix()
            for i in range(self.M):
                ## Evaluate total velocity components explicitly on the EXTERNAL face (per Hunt)
                u_hat = self.unit_normal_matrix[i]
                s_hat = self.unit_tangent_matrix[i]
                Vn_ext = np.dot(self.V_matrix[i], u_hat) + 0.5 * self.source_vector[i] ## This applies to each component check this jump condition (+ 0.5 * self.source_vector[i])
                Vs_ext = np.dot(self.V_matrix[i], s_hat) + 0.5 * self.vortex_vector[i] ## check this jump condition (0.5 * self.vortex_vector[i])
                ## Compute Hunt's normal boundary velocity error (Target - Computed) 
                epsilon_n = 0.0 - Vn_ext # 0.0 to remember that Hunt said epsilon_n = Vn-VnBC
                ## Calculate Hunt's relaxation targets
                sigma_target = -np.dot(u_hat, self.V_inf_vec) + epsilon_n # exactly as hunt defines it
                omega_target = Vs_ext - np.dot(s_hat, self.V_inf_vec) # exactly as hunt defines it
                ## 4. Residuals: current value minus Hunt's target destination
                self.Res_source_vector[i] = self.source_vector[i] - sigma_target
                self.Res_vortex_vector[i] = self.vortex_vector[i] - omega_target
            ## Apply relaxation step modifications
            self.vortex_vector -= (self.vortex_relaxation_factor * self.Res_vortex_vector) ## this applies to each component of vortex_vector
            self.source_vector -= (self.source_relaxation_factor * self.Res_source_vector) ## this applies to each component of source_vector
            ## Global Kutta condition trailing edge correction at lower (0) and upper (M-1) panels
            te_error = self.vortex_vector[self.M - 1] + self.vortex_vector[0]
            if abs(te_error) > self.acceptable_Kutta_error:
                self.vortex_vector -= 0.5 * te_error ## this applies to each component of vortex_vector
            ## Check maximum error bounds
            max_vortical_norm = np.linalg.norm(self.Res_vortex_vector, ord=np.inf)
            max_source_norm = np.linalg.norm(self.Res_source_vector, ord=np.inf)
            if ell % 10 == 0:
                print(f"Iteration {ell:3d} -> Max Vortex Res: {max_vortical_norm:.5e} | Max Source Res: {max_source_norm:.5e} | Trailing edge error: {te_error:.5e}")
            ell += 1

        print("Total iterations = ", ell, " out of ", self.max_relaxation_iterations)
        # print("\nvortex vector:\n", self.vortex_vector)
        # print("\nsource vector:\n", self.source_vector)
        # print("\nvelocity vector:\n", self.V_matrix)
        # print("\nsurface points:\n", self.nodes)

    def compute_pressure_coefficients(self):
        """
        Calculates the Pressure Coefficient (Cp) at the external face 
        of each panel using Bernoulli's equation.
        """
        V_inf_magnitude = np.linalg.norm(self.V_inf_vec)
        cp_array = np.zeros(self.M)
        for i in range(self.M):
            # Extract true external tangential velocity (baseline + half-jump)
            Vs_ext = np.dot(self.V_matrix[i], self.unit_tangent_matrix[i]) + 0.5 * self.vortex_vector[i]
            # Apply Bernoulli's equation: Cp = 1 - (V / V_inf)^2
            cp_array[i] = 1.0 - (Vs_ext / V_inf_magnitude) ** 2
        return cp_array
    
    def compute_lift_coefficient(self, cp_array, chord=1.0):
        """
        Integrates a given surface pressure distribution (Cp) across all 
        panels to calculate the total non-dimensional Lift Coefficient (CL).
        """
        total_force_x = 0.0
        total_force_y = 0.0
        # 1. Integrate normal pressure forces across the surface geometry
        for i in range(self.M):
            # Normal force magnitude (negative because pressure pushes inward)
            force_magnitude = -cp_array[i] * self.L_vals[i]
            # Decompose into global x and y components using panel normals
            total_force_x += force_magnitude * self.unit_normal_matrix[i, 0]
            total_force_y += force_magnitude * self.unit_normal_matrix[i, 1]
        # 3. Project global forces into the lift direction (perpendicular to freestream)
        lift = total_force_y * np.cos(self.alpha_rad) - total_force_x * np.sin(self.alpha_rad)
        # 4. Non-dimensionalize by chord to obtain Cl
        CL = lift / chord
        return CL
        
    def program(self, i):
        """This is a run function that uses all of the functions above."""
        self.pull_nodes(i)
        self.Argos_init()
        self.Argos_saver_scheme()
        cp_distribution = self.compute_pressure_coefficients()
        lift_coefficient = self.compute_lift_coefficient(cp_distribution, chord=1.0)
        print("CL: ", lift_coefficient)

if __name__ == "__main__":
    NACA_object = Relaxation("airfoils.json")
    NACA_object.program(0)  

        # def Argos_init(self):
    #     """Given the nodes and freestream velocity, compute unit tangent and unit normal vectors, Delta s, and K, and initialize sigma and omega for each control point"""
    #     ### Begin with initializing unit_tangent_matrix, unit_normal_matrix, Delta_s_vector, source_vector, and vortex_vector ### 
    #     ### Initialize numpy arrays for each of the matrices and vectors 
    #     self.unit_tangent_matrix = np.zeros((self.M,2))
    #     self.unit_normal_matrix = np.zeros((self.M,2))
    #     self.Delta_s_vector = np.zeros(self.M)
    #     self.source_vector = np.zeros(self.M)
    #     self.vortex_vector = np.zeros(self.M)
    #     ### loop through all control points ###
    #     for i in range(self.M):
    #         if (i==0):
    #             tangent_vector = self.control_points[i+1]-self.control_points[i]
    #             self.Delta_s_vector[i] = np.linalg.norm(tangent_vector)
    #         elif (i==self.N-1):
    #             tangent_vector = self.control_points[i]-self.control_points[i-1]
    #             self.Delta_s_vector[i] = np.linalg.norm(tangent_vector)
    #         else:
    #             tangent_vector = self.control_points[i+1]-self.control_points[i]
    #             p1 = self.control_points[i+1]-self.control_points[i]
    #             p2 = self.control_points[i]-self.control_points[i-1]
    #             self.Delta_s_vector[i] = 0.5*(np.linalg.norm(p1) + np.linalg.norm(p2))
    #         self.unit_tangent_matrix[i] = tangent_vector/np.linalg.norm(tangent_vector)
    #         normal_vector = np.array([-tangent_vector[1],tangent_vector[0]])
    #         self.unit_normal_matrix[i] = normal_vector/np.linalg.norm(normal_vector)
    #         self.source_vector[i] = -np.dot(self.unit_normal_matrix[i],self.V_inf_vec)
    #         self.vortex_vector[i] = self.initial_vortex
    #     # now initialize the Kernel "K" matrix, which is NxN. Each entry is a 2-D vector. 
    #     self.init_Kernel()

    # def init_Kernel(self):
    #     """creates the kernel matrix"""
    #     self.Kernel = np.zeros((self.N,self.N, 2)) # NxN matrix where each entry is a 2D vector.
    #     for i in range(self.N):
    #         P = self.control_points[i]
    #         for j in range(self.N):
    #             Q = self.control_points[j]
    #             r_QP = P-Q
    #             if i != j:
    #                 self.Kernel[i][j] = r_QP/(2*np.pi*np.dot(r_QP,r_QP))
    #             else:
    #                 self.Kernel[i][j] = r_QP
    #     return self.Kernel    

    # def calc_v_matrix(self):
    #     """"""  
    #     for i in range(self.N):
    #         self.V_matrix[i] = self.V_inf_vec
    #         for j in range(self.N):
    #             Delta = self.Delta_s_vector[j]
    #             Kernel_first = self.Kernel[i][j][0]
    #             Kernel_second = self.Kernel[i][j][1]
    #             source = self.source_vector[j] 
    #             vortex = self.vortex_vector[j]
    #             if i==j:
    #                 self.V_matrix[i] += 0.5*source*self.unit_normal_matrix[j]
    #             else:
    #                 self.V_matrix[i][0] += (source*Kernel_first + vortex*Kernel_second)*Delta ### using first component of cross product
    #                 self.V_matrix[i][1] += (source*Kernel_second - vortex*Kernel_first)*Delta 
    #     return self.V_matrix

    # def Argos_saver_scheme(self):
    #     """"""
    #     ell = 0 # initialize count
    #     max_vortical_norm = 100 # initialize vortex residual
    #     max_source_norm = 100 # initialize source residual
    #     self.Res_vortex_vector = np.zeros((self.N))
    #     self.Res_source_vector = np.zeros((self.N))
    #     self.V_matrix = np.zeros((self.N,2)) # initialize velocity vector 
    #     while ((max_vortical_norm > self.acceptable_vort_error) or (max_source_norm > self.acceptable_source_error)) and (ell < self.max_relaxation_iterations):
    #         self.calc_v_matrix()
    #         for i in range(self.N):
    #             Vn = np.dot(self.V_matrix[i],self.unit_normal_matrix[i])
    #             # print("Vn: ", Vn)
    #             Vs = np.dot(self.V_matrix[i],self.unit_tangent_matrix[i])
    #             # print("Vs: ", Vs)
    #             self.Res_vortex_vector[i] = self.vortex_vector[i]-(Vs-np.dot(self.unit_tangent_matrix[i],self.V_inf_vec)) 
    #             self.Res_source_vector[i] = self.source_vector[i]-(Vn-np.dot(self.unit_normal_matrix[i],self.V_inf_vec)) 
            
    #         i = 0
    #         for i in range(self.N):
    #             self.vortex_vector[i] -= (self.vortex_relaxation_factor*self.Res_vortex_vector[i])
    #             self.source_vector[i] -= (self.source_relaxation_factor*self.Res_source_vector[i])
            
    #         if abs(self.vortex_vector[self.N-1]+self.vortex_vector[0]) > self.acceptable_Kutta_error:
    #             for i in range(self.N):
    #                 self.vortex_vector[i] -= 0.5*(self.vortex_vector[self.N-1]+self.vortex_vector[0])
            
    #         max_vortical_norm = np.linalg.norm(self.Res_vortex_vector, ord=np.inf)
    #         max_source_norm = np.linalg.norm(self.Res_source_vector,ord=np.inf)
    #         ell += 1

    # def calc_v_matrix(self):
    #     """Calculates total velocity at each control point including external face self-induction"""  
    #     for i in range(self.M):
    #         self.V_matrix[i] = self.V_inf_vec.copy()
    #         for j in range(self.M):
    #             Delta = self.Delta_s_vector[j]
    #             Kernel_first = self.Kernel[i][j][0]
    #             Kernel_second = self.Kernel[i][j][1]
    #             source = self.source_vector[j] 
    #             vortex = self.vortex_vector[j]
                
    #             if i == j:
    #                 # SAVER external-face jump conditions: normal from source, tangential from vortex
    #                 self.V_matrix[i] += 0.5 * source * self.unit_normal_matrix[j] + 0.5 * vortex * self.unit_tangent_matrix[j]
    #             else:
    #                 self.V_matrix[i][0] += (source * Kernel_first + vortex * Kernel_second) * Delta 
    #                 self.V_matrix[i][1] += (source * Kernel_second - vortex * Kernel_first) * Delta 
    #     return self.V_matrix

    # def init_Kernel(self):
    #     """Creates the kernel matrix of size M x M"""
    #     self.Kernel = np.zeros((self.M, self.M, 2)) 
    #     for i in range(self.M):
    #         P = self.control_points[i]
    #         for j in range(self.M):
    #             Q = self.control_points[j]
    #             r_QP = P - Q
    #             if i != j:
    #                 self.Kernel[i][j] = r_QP / (2 * np.pi * np.dot(r_QP, r_QP))
    #             else:
    #                 # Self-influence handled analytically in calc_v_matrix, set to zero here
    #                 self.Kernel[i][j] = np.zeros(2)
    #     return self.Kernel 