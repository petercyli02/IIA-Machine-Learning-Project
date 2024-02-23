
"""
fork from python-rl and pybrain for visualization
"""

import numpy as np
import matplotlib.pyplot as plt

# If theta  has gone past our conceptual limits of [-pi,pi]
# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
     
def remap_angle(theta):
    return (theta - np.pi) % (2*np.pi) - np.pi


## loss function given a state vector. the elements of the state vector are
## [cart location, cart velocity, pole angle, pole angular velocity]

def loss(state):
    
    sigma_l = 0.5
    return 1 - np.exp( - state @ state / (2.0 * sigma_l**2) )


class CartPole:
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """

    def __init__(self, visual=False, smooth=False, save_frames=False, fig_num=1):
        
        self.reset() 
        
        self.visual = visual
        self.save_frames = save_frames
        self.frame = 0

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        # or they vary by some scalar from the other poles)
        self.pole_length = 0.5 
        self.pole_mass = 0.5 

        self.mu_c = 0.001 # friction coefficient of the cart
        self.mu_p = 0.001 # friction coefficient of the pole
        self.sim_steps = 50 # number of Euler integration steps to perform in one go
        self.delta_time = 0.2 # time step of the Euler integrator 
        self.max_force = 20.
        self.gravity = 9.8
        self.cart_mass = 0.5

        # set the euler integration settings to smaller steps to allow smooth rendering
        if smooth:
            self.sim_steps = 20
            self.delta_time = 0.08
           
        # for plotting
        self.cartwidth = 1.0
        self.cartheight = 0.2
    
        if self.visual:
            self.drawPlot( fig_num )
 

    # reset the state vector to the initial state (down-hanging pole)
    def reset(self):
        self.set_state( [ 0, 0, np.pi, 0 ] ) 
            
    def set_state(self, state):
        
        self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angvel = state[:4]
           
            
    def get_state(self):

        return np.array([ self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angvel ])

    
    def remap_angle(self):
        
        self.pole_angle = remap_angle( self.pole_angle )
   

    # the loss function that the policy will try to optimise (lower) as a member function
    def loss(self):
        return loss(self.getState())
    
        
    # This is where the equations of motion are implemented
    def perform_action( self, action=0.0 ):
        
        # prevent the force from being too large
        force = self.max_force * np.tanh(action/self.max_force)
        dt = self.delta_time / float(self.sim_steps)
 
        # integrate forward the equations of motion using the Euler method
        for step in range(self.sim_steps):

            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            
            m = 4.0 * ( self.cart_mass + self.pole_mass ) - 3.0 * self.pole_mass * c**2
            
            cart_accel = 1/m * ( 
                2.0 * ( self.pole_length * self.pole_mass * s * self.pole_angvel**2
                + 2.0 * ( force - self.mu_c * self.cart_velocity ) )
                - 3.0 * self.pole_mass * self.gravity * c*s 
                + 6.0 * self.mu_p * self.pole_angvel * c / self.pole_length
            ) 
            
            pole_accel = 1/m * (
                - 3.*c * 2. / self.pole_length * ( 
                    self.pole_length / 2.0 * self.pole_mass * s * self.pole_angvel**2 
                    + force 
                    - self.mu_c * self.cart_velocity
                )
                + 6.0 * ( self.cart_mass + self.pole_mass ) / ( self.pole_mass * self.pole_length ) * \
                ( self.pole_mass * self.gravity * s - 2.0/self.pole_length * self.mu_p * self.pole_angvel )
            )
           
            # Update state variables
            # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
            self.cart_velocity += dt * cart_accel
            self.pole_angvel   += dt * pole_accel
            self.pole_angle    += dt * self.pole_angvel
            self.cart_position += dt * self.cart_velocity

        if self.visual:
            self._render()

            
    # the following are graphics routines
    def drawPlot(self, fig_num):
        
        plt.ion()
        self.fig, self.axes = plt.subplots( 1, 1, num=fig_num, figsize=(9.5,2) )
        
        # draw cart
        self.axes.get_yaxis().set_visible(False)
        self.box = plt.Rectangle(xy=(self.cart_position - self.cartwidth / 2.0, -self.cartheight / 2.0), 
                             width=self.cartwidth, height=self.cartheight)
        self.axes.add_artist(self.box)
        self.box.set_clip_box(self.axes.bbox)

        # draw pole
        self.pole = plt.Line2D([self.cart_position, self.cart_position + np.sin(self.pole_angle) * self.pole_length], 
                           [0, np.cos(self.pole_angle) * self.pole_length], linewidth=3.5, color='black')
        self.axes.add_artist(self.pole)
        self.pole.set_clip_box(self.axes.bbox)

        # set axes limits
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-1, 1)
        #self.fig.tight_layout()
 
        self.fig.subplots_adjust( top=0.92, bottom=0.17, left=0.02, right=0.98 )
        
    def _render(self):
        
        self.box.set_x(self.cart_position - self.cartwidth / 2.0)
        self.pole.set_xdata([ self.cart_position, self.cart_position + np.sin(self.pole_angle) * self.pole_length ])
        self.pole.set_ydata([ 0, np.cos(self.pole_angle) * self.pole_length ])

        self.fig.canvas.draw()
        
        if self.save_frames:
            self.fig.savefig( f"frames/{self.frame}.png", dpi=300 )
            self.frame += 1

class Object(object):
    pass 

# static version of perform action
def perform_action( state, action=0.0, better_angle=False ):

    self = Object()
   
    self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angvel = state
    
    if better_angle:
        self.pole_angle += np.pi

    self.pole_length = 0.5 
    self.pole_mass = 0.5 

    self.mu_c = 0.001 # friction coefficient of the cart
    self.mu_p = 0.001 # friction coefficient of the pole
    self.sim_steps = 50 # number of Euler integration steps to perform in one go
    self.delta_time = 0.2 # time step of the Euler integrator 
    self.max_force = 20.
    self.gravity = 9.8
    self.cart_mass = 0.5

    # prevent the force from being too large
    force = self.max_force * np.tanh(action/self.max_force)
    dt = self.delta_time / float(self.sim_steps)

    # integrate forward the equations of motion using the Euler method
    for step in range(self.sim_steps):

        s = np.sin(self.pole_angle)
        c = np.cos(self.pole_angle)

        m = 4.0 * ( self.cart_mass + self.pole_mass ) - 3.0 * self.pole_mass * c**2

        cart_accel = 1/m * ( 
            2.0 * ( self.pole_length * self.pole_mass * s * self.pole_angvel**2
            + 2.0 * ( force - self.mu_c * self.cart_velocity ) )
            - 3.0 * self.pole_mass * self.gravity * c*s 
            + 6.0 * self.mu_p * self.pole_angvel * c / self.pole_length
        ) 

        pole_accel = 1/m * (
            - 3*c * 2/self.pole_length * ( 
                self.pole_length/2 * self.pole_mass * s * self.pole_angvel**2 
                + force 
                - self.mu_c * self.cart_velocity
            )
            + 6.0 * ( self.cart_mass + self.pole_mass ) / ( self.pole_mass * self.pole_length ) * \
            ( self.pole_mass * self.gravity * s - 2.0/self.pole_length * self.mu_p * self.pole_angvel )
        )

        # Update state variables
        # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
        self.cart_velocity += dt * cart_accel
        self.pole_angvel   += dt * pole_accel
        self.pole_angle    += dt * self.pole_angvel
        self.cart_position += dt * self.cart_velocity
        
    if better_angle:
        self.pole_angle -= np.pi

    return np.array( [ self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angvel ] )


def perform_action5( state5 ):
    
    self = Object()
   
    self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angvel, action = state5
    
    self.pole_length = 0.5 
    self.pole_mass = 0.5 

    self.mu_c = 0.001 # friction coefficient of the cart
    self.mu_p = 0.001 # friction coefficient of the pole
    self.sim_steps = 500 # number of Euler integration steps to perform in one go
    self.delta_time = 0.2 # time step of the Euler integrator 
    self.max_force = 20.
    self.gravity = 9.8
    self.cart_mass = 0.5

    # prevent the force from being too large
    force = self.max_force * np.tanh(action/self.max_force)
    dt = self.delta_time / float(self.sim_steps)

    # integrate forward the equations of motion using the Euler method
    for step in range(self.sim_steps):

        s = np.sin(self.pole_angle)
        c = np.cos(self.pole_angle)

        m = 4.0 * ( self.cart_mass + self.pole_mass ) - 3.0 * self.pole_mass * c**2

        cart_accel = 1/m * ( 
            2.0 * ( self.pole_length * self.pole_mass * s * self.pole_angvel**2
            + 2.0 * ( force - self.mu_c * self.cart_velocity ) )
            - 3.0 * self.pole_mass * self.gravity * c*s 
            + 6.0 * self.mu_p * self.pole_angvel * c / self.pole_length
        ) 

        pole_accel = 1/m * (
            - 3*c * 2/self.pole_length * ( 
                self.pole_length/2 * self.pole_mass * s * self.pole_angvel**2 
                + force 
                - self.mu_c * self.cart_velocity
            )
            + 6.0 * ( self.cart_mass + self.pole_mass ) / ( self.pole_mass * self.pole_length ) * \
            ( self.pole_mass * self.gravity * s - 2.0/self.pole_length * self.mu_p * self.pole_angvel )
        )

        # Update state variables
        # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
        self.cart_velocity += dt * cart_accel
        self.pole_angvel   += dt * pole_accel
        self.pole_angle    += dt * self.pole_angvel
        self.cart_position += dt * self.cart_velocity
        
    return np.array( [ self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angvel, action ] )
    


pole_length = 0.5 
pole_mass   = 0.5 
mu_c        = 0.001
mu_p        = 0.001
max_force   = 20.
gravity     = 9.8
cart_mass   = 0.5


def perform_action_RK4( state ):
    
    h = 0.1 # RK4 step size
      
    # perform 2 RK4 steps
    for _ in range(2):
        
        k1 = dstate_dt( state            )
        k2 = dstate_dt( state + h/2 * k1 ) 
        k3 = dstate_dt( state + h/2 * k2 ) 
        k4 = dstate_dt( state + h   * k3 ) 

        state = state + h/6 * ( k1 + 2*k2 + 2*k3 + k4 ) 

    return state


def dstate_dt( state ):

    cart_position, \
    cart_velocity, \
    pole_angle,    \
    pole_angvel,   \
    action = state
    
    dstate = np.array( [0., 0, 0, 0, 0] )
    
    dstate[0] = state[1]
    dstate[2] = state[3]
  

    force = max_force * np.tanh( action / max_force )

    s = np.sin(pole_angle)
    c = np.cos(pole_angle)

    m = 4.0 * ( cart_mass + pole_mass ) - 3.0 * pole_mass * c**2
    
    dstate[1] = 1/m * (
        2.0 * ( pole_length * pole_mass * s * pole_angvel**2
        + 2.0 * ( force - mu_c * cart_velocity ) )
        - 3.0 * pole_mass * gravity * c*s 
        + 6.0 * mu_p * pole_angvel * c / pole_length
    ) 

    dstate[3] = 1/m * (
        - 3.*c * 2./pole_length * ( 
            pole_length/2. * pole_mass * s * pole_angvel**2 
            + force 
            - mu_c * cart_velocity
        )
        + 6.0 * ( cart_mass + pole_mass ) / ( pole_mass * pole_length ) * \
        ( pole_mass * gravity * s - 2.0/pole_length * mu_p * pole_angvel )
    )

    return dstate
