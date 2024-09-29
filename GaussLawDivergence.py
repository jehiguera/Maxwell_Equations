
"""
Maxwell equations. Gauss Law:  Maxwell Equation for electric fields in differential mode definition.

The electric field produced by electric charge diverges from positive charge and converges upon negative charge

The differential form of Gauss law is useful in any problem in which the spatial variation of the vector 
electric field is known at a specified location. In this case you can find the volume charge density at that location 
using this equation in differential format and if the volume charge density is known, the divergence of the electric 
field may be determined.


Author: Jorge Higuera Portilla
email:  Jhiguera@ieee.org
Date:   29/09/2024

"""
#Python Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import tools 3D  Matplotlib
from scipy.ndimage import gaussian_filter  # Import gaussian filter to improve 3D figure view

class Gauss:

    """
    The Gauss class
    """
    E0 = 1.0  # Constant initial Electric field. units V/m

    def __init__(self, E0=1):

        """
        Constructor Gauss class.
    
        Args:
        E0 (float): Initial electric field. Units: V/m. default value: 1.0 V/m.
        """
        self.E0 = E0
    
    @staticmethod
    def gridLimits(init, final, my_range):
        """
        Method to define an array containing  my_range of values with and init and final value. 
        Create a set number of evenly spaced points within a specific interval.

        Args:
        init (float): The starting value of the range 
        final (float): The ending value of the range.
        my_range (int): The number of points to generate within this range

        Returns:
        tuple: two arrays linespace in x and y.

        """
        if not isinstance(my_range, int) or my_range <= 0:
            raise ValueError("Error!!! my_range must be a positive integer.")
        
        x = np.linspace(init, final, my_range)
        y = np.linspace(init, final, my_range)
        
        return x, y
    
    @staticmethod
    def mymeshgrid(x,y):

        """"
        Meshgrid 2D definition

        Args:
         x (array): linespace in x
         y (array): linespace in y

        Returns
        tuple: meshgrid 2D.
        """
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays x and y cannot be empty.")
    
        X, Y = np.meshgrid(x, y)
        return X, Y
    
    def electricFieldVector(self, line_x, line_y, verbose=False):
        """
        # This method calculate the components of the electric field normalized and adimensional.
        # Note: To represent a real electric field in units of V/m, these expressions need to be
        # multiplied by a constant E_0 representing the field's amplitude for example E0 = 1V/m.

        Args:
        line_x (array):  linespace array in x axis
        line_y (array):  linespace array in y axis
        E0: constant E_0 representing the field's amplitude for example E0 = 1V/m.
        verbose (boolean): False by default. Optional True to print Ex and Ey


        Returns:
        tuple: coordinates of electric fields Ex and Ey 
        """
        if line_x.shape != line_y.shape:
            raise ValueError("Error!! line_x and line_y must have the same shape.")


        Ex = self.E0*np.sin(np.pi/2 * line_y)  
        Ey = -self.E0*np.sin(np.pi/2 * line_x)
        
        if verbose:
            print(f"Electric field Ex units (V/m): {Ex}")  
            print(f"Electric field Ey units (V/m): {Ey}")
        
        return Ex, Ey
    
    def gradientElectricField(self, Efield_x, line_x, Efield_y, line_y, verbose=False):
        """"
        Computes the partial derivatives of the electric field components
        
        Args:
        Efield_x (array): x-component of the electric field
        line_x (array):  Linespace array in x-axis
        Efield_y (array): y-component of the electric field
        line_y (array):  Linespace array in y-axis
        verbose (boolean): False by default. Optional True to print dEx_dx and dEy_dy

        Returns:
        tuple: partial derivatives of electric fields (dEx_dx, dEy_dy)
        """
        if Efield_x.shape[1] != len(line_x):
            raise ValueError("The number of columns in Efield_x must match the length of line_x.")
        if Efield_y.shape[0] != len(line_y):
            raise ValueError("The number of rows in Efield_y must match the length of line_y.")
    
        dEx_dx = np.gradient(Efield_x, line_x, axis=1)  # partial derivative Ex respect to x
        dEy_dy = np.gradient(Efield_y, line_y, axis=0)  # partial derivative Ey respect to y

        if verbose:
            print("dEx_dx (V/m^2):", dEx_dx)
            print("dEy_dy (V/m^2):", dEy_dy)
        
        return dEx_dx, dEy_dy
    
    def divergence_E(self, der_Ex_dx, der_Ey_dy, verbose= False):
        """"
        Computes the divergence of the electric field
        
        Args:
        der_Ex_dx (array): partial derivatives of the electric field in the x-direction
        der_Ey_dy (array): partial derivatives of the electric field in the y-direction

        Returns:
        tuple: Divergence of the electric field, absolute value of the divergence, and its maximum value.
        """
        div_E = der_Ex_dx + der_Ey_dy
        
        abs_div_E = np.abs(div_E)
        max_abs_max_div_E = np.round(np.max(abs_div_E), 16)  #Rounds to 16 decimal places
        
        if verbose:
            print("Divergence absolute value (V/m^2):", abs_div_E)
            print("divergence maximum value (V/m^2):", max_abs_max_div_E)
        
        return div_E, abs_div_E, max_abs_max_div_E
    
    def figure_Div_E(self, X, Y, Ex, Ey, div_E, max_abs_max_div_E):
        
        """"
        Generates a 2D quiver plot of the electric field and its divergence.

        Args:
        
        X, Y (array): Meshgrid coordinates.
        Ex, Ey (array): Components of the electric field.
        div_E (array): Divergence of the electric field.
        max_abs_max_div_E (float): Maximum absolute value of the divergence.
        
        """
        
        plt.figure(figsize=(8, 6))
        plt.quiver(X, Y, Ex, Ey, div_E, scale=10, cmap='coolwarm')
        plt.colorbar(label=f"Divergence E_max value:  {max_abs_max_div_E} (V/m^2)")
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Maxwell equation: Gauss Law. Divergence of electric field E')
        plt.show()
        return

    def figure_Div_E_3D(self, X, Y, div_E, max_abs_max_div_E):
        """
        Generates a 3D surface plot of the divergence of the electric field.

        Args:
        X, Y (array): Meshgrid coordinates.
        div_E (array): Divergence of the electric field.
        max_abs_max_div_E (float): Maximum absolute value of the divergence.
        """
        # smooth surface  gaussian filter
        div_E_smooth = gaussian_filter(div_E, sigma=1)
    
        # Create a new figure with smaller size
        fig = plt.figure(figsize=(6, 4), dpi=180)  # Ajustar 'figsize' y 'dpi' para reducir el tamaño de la figura
    
        # Add a 3D axis to the figure
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
        # Plot the surface
        surf = ax.plot_surface(X, Y, div_E_smooth, cmap='coolwarm', edgecolor='none', rstride=1, cstride=1)
    
        # Add a color bar for reference; link it to the surface 'surf'
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label(f"Divergence: {max_abs_max_div_E} (V/m^2)", fontsize=6)
        cbar.ax.tick_params(labelsize=6)  # Adjust size numbers scale color bar
        
        # adjust of  offset text size and format
        cbar.ax.yaxis.get_offset_text().set_fontsize(6)
        cbar.formatter.set_powerlimits((0, 0))  # bar format consistent
        cbar.update_normal(surf)
    
        # Set labels
        ax.set_xlabel('x (m)', fontsize=6)
        ax.set_ylabel('y (m)', fontsize=6)
        ax.set_zlabel('Divergence (V/m^2)', fontsize=6)

         # Adjust label scale
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='z', labelsize=6)
    
        # Set the main title using 'fig.suptitle' for better control
        fig.suptitle(f' Divergence of Electric Field (V/m^2)', fontsize=8)

        # Adjust spacing around the plot
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    
        # Show the plot
        plt.show()

    def figure_Div_E_3D_vectors(self, X, Y, Ex, Ey, div_E, max_abs_max_div_E):
        """
        Generates a 3D quiver plot of the electric field vectors with colors indicating divergence.

        Args:
        X, Y (array): Meshgrid coordinates.
        Ex, Ey (array): Components of the electric field.
        div_E (array): Divergence of the electric field.
        max_abs_max_div_E (float): Maximum absolute value of the divergence.
        """
        # figure  3D
        fig = plt.figure(figsize=(6, 4), dpi=180)
        ax = fig.add_subplot(111, projection='3d')
        
        # Coordinates Z  flags plane XY
        Z = np.zeros_like(X)
        
        # Component Z vector (W)
        W = np.zeros_like(Ex)
        
        # Normalization values divergence  color map
        norm = plt.Normalize(div_E.min(), div_E.max())
        
        # 3D flags individual
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                color = plt.cm.coolwarm(norm(div_E[i, j]))  # Obtener el color para la divergencia actual
                ax.quiver(X[i, j], Y[i, j], Z[i, j], Ex[i, j], Ey[i, j], W[i, j], 
                        length=0.1, color=color, normalize=True)
        
        # color bar divergence
        mappable = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        mappable.set_array(div_E)
        fig.colorbar(mappable, ax=ax, label='Divergence (V/m^2)')
        
       
        ax.set_xlabel('x (m)', fontsize=6)
        ax.set_ylabel('y (m)', fontsize=6)
        ax.set_zlabel('z (m)', fontsize=6)
        
        
        ax.set_title(f'Electric Field Vectors\nMax Abs Divergence: {max_abs_max_div_E} (V/m^2)', fontsize=8)
        
        
        plt.show()



# Example of use: 
# 1. Define an instance of Gauss object with an initial Electric field E0. By default E0 = 1V/m
particle1= Gauss()
particle2= Gauss(E0=10)

#2. Define the position of the particles
x1, y1 = particle1.gridLimits(-0.5, 0.5, 10)
x2, y2 = particle2.gridLimits(-0.1, 0.1, 20)

#3. Creation of the meshgrid in 2 dimensions
X1, Y1 = particle1.mymeshgrid(x1,y1)
X2, Y2 = particle2.mymeshgrid(x2,y2)


#4. Define the Components of the electric field in x and y directions. 
# Units: newtons per coulom (N/C) that is equivalent to volts per meter (V/m)
Ex1, Ey1 = particle1.electricFieldVector(X1,Y1, verbose=True)
Ex2, Ey2 = particle2.electricFieldVector(X2,Y2, verbose=True)

#5. Compute the partial derivated of electric field. Units V/m^2
dEx_dx1, dEy_dy1 = particle1.gradientElectricField(Ex1, x1, Ey1, y1, verbose=True)
dEx_dx2, dEy_dy2 = particle2.gradientElectricField(Ex2, x2, Ey2, y2, verbose=True)

#6. Compute the Divergence of electric field, max value, abs max value. Units: Volts/meter^2 (V/m^2)
div_E1, abs_div_E1, max_abs_max_div_E1 = particle1.divergence_E(dEx_dx1,dEy_dy1, verbose=True)
div_E2, abs_div_E2, max_abs_max_div_E2 = particle2.divergence_E(dEx_dx2,dEy_dy2, verbose=True)

#7. Generates a 2D quiver plot of the electric field and its divergence
particle1.figure_Div_E(X1, Y1, Ex1, Ey1, div_E1, max_abs_max_div_E1)
#particle2.figure_Div_E(X2, Y2, Ex2, Ey2, div_E2, max_abs_max_div_E2)

#8 Generates a 3D representation of electric field and divergence
particle1.figure_Div_E_3D(X1, Y1, div_E1, max_abs_max_div_E1)

particle1.figure_Div_E_3D_vectors(X1,Y1,Ex1,Ey1,div_E1,max_abs_max_div_E1)