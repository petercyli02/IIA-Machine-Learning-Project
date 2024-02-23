# 1.3 - predicted next step

class LinearModel:
    def __init__(self, number_of_points=500, delta_time=0.2, sim_steps=50, start_state=None,generate_next_state_randomly=True, 
                 x_range=(-10, 10), theta_range=(-np.pi, np.pi), 
                 x_dot_range=(-10, 10), theta_dot_range=(-15, 15), remap_angle=False):

        self.delta_time = delta_time
        self.sim_steps = sim_steps
        self.number_of_points = number_of_points
        self.generate_next_state_randomly = generate_next_state_randomly
        self.x_range = x_range
        self.x_dot_range = x_dot_range
        self.theta_range = theta_range
        self.theta_dot_range = theta_dot_range
        self.start_state = start_state
        self.remap_angle = remap_angle
        self.model = self.get_model()
        
    
    def random_state(self):
        return np.array([np.random.uniform(*self.x_range), np.random.uniform(*self.x_dot_range), 
                         np.random.uniform(*self.theta_range), np.random.uniform(*self.theta_dot_range)])
    
    def get_data_itertatively(self, start_state):
        past_state = np.zeros((self.number_of_points, 4))
        future_state = np.zeros((self.number_of_points, 4))
        cartpole = CartPole()
        cartpole.delta_time = self.delta_time
        cartpole.sim_steps = self.sim_steps
        for i in range(self.number_of_points):
            current_state = start_state if i == 0 else future_state[i-1] # The current state is the previous future state
            if self.remap_angle:
                current_state[2] = remap_angle(current_state[2])
            past_state[i] = current_state
            cartpole.setState(past_state[i])
            cartpole.performAction(0)
            new_state = cartpole.getState()
            if self.remap_angle:
                new_state[2] = remap_angle(new_state[2])
            future_state[i] = new_state
        
        change = future_state - past_state
        return past_state, change


    def get_data_randomly(self):
        past_state = np.zeros((self.number_of_points, 4))
        future_state = np.zeros((self.number_of_points, 4))
        np.random.seed(0)
        cartpole = CartPole()
        cartpole.delta_time = self.delta_time
        cartpole.sim_steps = self.sim_steps
        for i in range(self.number_of_points):
            cartpole.setState(self.random_state())
            current_state = cartpole.getState()
            if self.remap_angle:
                current_state[2] = remap_angle(current_state[2])
            past_state[i] = current_state
            cartpole.performAction(0)
            new_state = cartpole.getState()
            if self.remap_angle:
                new_state[2] = remap_angle(new_state[2])
            future_state[i] = new_state
        change = future_state - past_state 
        return past_state, change
        
    def get_data(self):
        if self.generate_next_state_randomly:
            print("Generating data randomly")
            return self.get_data_randomly()
        else:
            return self.get_data_itertatively(self.start_state)

    def get_model(self):
        past_state, change = self.get_data()
        C = np.linalg.lstsq(past_state, change, rcond=None)[0]
        return C
    

    def train_model(self, past_state, change):
        self.model = np.linalg.lstsq(past_state, change, rcond=None)[0]
        return self.model
    
    def predict_change(self, state):
        change = state @ self.model
        return change

    def predict_state(self, state, remap_theta=True):
        change = self.predict_change(state)
        new_state =  state + change
        if remap_theta:
            new_state[2] = remap_angle(new_state[2])
        return new_state
    
    def save_model(self, path):
        np.save(path, self.model)

small_time_step_model = LinearModel(number_of_points=500, delta_time=0.2, sim_steps=50, generate_next_state_randomly=True)
print(small_time_step_model.model)
past_state, change = small_time_step_model.get_data_randomly()
future = past_state + change
model_predictions = small_time_step_model.predict_change(past_state)
predictions = model_predictions.T
future_predictions = past_state + model_predictions
fig, ax = plt.subplots(2, 2, figsize=(20, 10))

ax[0, 0].scatter(future[:, 0], future_predictions[:, 0], label='x')
ax[0, 0].set_xlabel('Actual state')
ax[0, 0].set_ylabel('Predicted state')
ax[0, 0].title.set_text('Cart location')

ax[0, 1].scatter(future[:, 1], future_predictions[:, 1], label='x_dot')
ax[0, 1].set_xlabel('Actual state')
ax[0, 1].set_ylabel('Predicted state')
ax[0, 1].title.set_text('Cart velocity')

ax[1, 0].scatter(future[:, 2], future_predictions[:, 2], label='theta')
ax[1, 0].set_xlabel('Actual state')
ax[1, 0].set_ylabel('Predicted state')
ax[1, 0].title.set_text('Pole angle')

ax[1, 1].scatter(future[:, 3], future_predictions[:, 3], label='theta_dot')
ax[1, 1].set_xlabel('Actual state')
ax[1, 1].set_ylabel('Predicted state')
ax[1, 1].title.set_text('Pole velocity')

fig.suptitle('Predicted state vs Actual state after one call to performAction()')
plt.show()

# Plotting the real values against the predicted values
plt.rcParams["figure.figsize"] = (20, 10)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle("Actual change in state vs Predicted change in state after one call to performAction()", fontsize=20)
ax1.scatter(change[:, 0], predictions[0])
#ax1.plot([-2, 2], [-2, 2], color='red', label='y=x', linewidth=5)
ax1.set_title("Cart location", fontsize=20)
ax1.legend(fontsize=15)

ax2.scatter(change[:, 1], predictions[1])
ax2.set_title("Cart velocity", fontsize=20)

ax3.scatter(change[:, 2], predictions[2])
#ax3.plot([-3, 3], [-3, 3], color='red', label='y=x', linewidth=5)
ax3.set_title("Pole angle", fontsize=20)
ax3.legend(fontsize=15)

ax4.scatter(change[:, 3], predictions[3])
ax4.set_title("Pole velocity", fontsize=20)
for ax in (ax1, ax2, ax3, ax4):
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("Actual change", fontsize=15)
    ax.set_ylabel("Predicted change", fontsize=15)
plt.tight_layout()
plt.show()
plt.rcParams["figure.figsize"] = (6.4, 4.8)

class SingleVariableScan:
    def __init__(self, num_search_points, x_constant, x_dot_constant, theta_constant, theta_dot_constant, model,
                 x_range=(-1, 1), x_dot_range=(-10, 10), theta_range=(-np.pi, np.pi), theta_dot_range=(-15, 15), delta_time=0.2, sim_steps=50):
        # Initialise the system based on the constant initial conditions
        initial_state_case_1 = np.ones((4, num_search_points))
        initial_state_case_1[0] *= x_constant
        initial_state_case_1[1] *= x_dot_constant
        initial_state_case_1[2] *= theta_constant
        initial_state_case_1[3] *= theta_dot_constant
        initial_state = np.stack([initial_state_case_1] * 4, axis=0)
        # For each of the 4 cases, vary one of the state variables with the appropriate range and keep the others constant 
        initial_state[0, 0, :] = np.linspace(x_range[0], x_range[1], num_search_points)
        initial_state[1, 1, :] = np.linspace(x_dot_range[0], x_dot_range[1], num_search_points)
        initial_state[2, 2, :] = np.linspace(theta_range[0], theta_range[1], num_search_points)
        initial_state[3, 3, :] = np.linspace(theta_dot_range[0], theta_dot_range[1], num_search_points)
        # Store the initial state and the change in state
        self.model = model
        self.initial_state = initial_state
        self.change_in_state = np.zeros_like(initial_state)
        self.model_change_in_state = np.zeros_like(initial_state)
        self.number_of_search_points = num_search_points
        self.delta_time = delta_time
        self.sim_steps = sim_steps

    def compute_change(self):
        cartpole = CartPole()
        cartpole.delta_time = self.delta_time
        cartpole.sim_steps = self.sim_steps
        for i in range(4):
            for j in range(self.number_of_search_points):
                cartpole.setState(self.initial_state[i, :, j])    
                cartpole.performAction(0)
                self.change_in_state[i, :, j] = cartpole.getState() - self.initial_state[i, :, j]
                self.model_change_in_state[i, :, j] = self.model.predict_change(self.initial_state[i, :, j])



    def plot_change(self, fig=None, axs=None, case="", linestyle1="solid", linestyle2="solid"):
        if axs is None:
            fig, axs = plt.subplots(4, 4)
        fig.suptitle('Change in state variables when one state variable is varied', fontsize=30, y = 1.5)
        state_variables = ["x", "$\\dot{x}$", "$\\theta$", "$\\dot{\\theta}$"]
        for i in range(4):
            x_axis = self.initial_state[i, i, :]
            for j in range(4):
                y_actual = self.change_in_state[i, j, :]
                y_predicted = self.model_change_in_state[i, j, :]
                axs[i, j].plot(x_axis, y_actual, label="Actual Change", linewidth=5, linestyle=linestyle1)
                axs[i, j].plot(x_axis, y_predicted, label="Predicted Change", linewidth=5, linestyle=linestyle2)
                axs[i, j].set_title(f"$\\Delta$ {state_variables[j]}", fontsize=20)
                axs[i, j].set_xlabel(state_variables[i], fontsize=20)
                axs[i, j].set_ylabel(None, fontsize=20)
                axs[i, j].tick_params(axis='both', which='major', labelsize=15)
                #axs[i, j].legend(fontsize=20)
        return axs
    
    def plot_difference_in_change(self, fig=None, axs=None, case="", linestyle="solid"):
        if axs is None:
            fig, axs = plt.subplots(4, 4)
        fig.suptitle('Difference between Actual Change and Pricted Change', fontsize=30)
        state_variables = ["x", "$\\dot{x}$", "$\\theta$", "$\\dot{\\theta}$"]
        for i in range(4):
            x_axis = self.initial_state[i, i, :]
            for j in range(4):
                y_actual = self.change_in_state[i, j, :]
                y_predicted = self.model_change_in_state[i, j, :]
                difference = y_actual - y_predicted
                axs[i, j].plot(x_axis, difference, label="Difference" + case, linewidth=5, linestyle=linestyle)
                axs[i, j].set_title(f"$\\Delta$ {state_variables[j]}", fontsize=20)
                axs[i, j].set_xlabel(state_variables[i], fontsize=20)
                axs[i, j].set_ylabel(None, fontsize=20)
                axs[i, j].tick_params(axis='both', which='major', labelsize=15)
                #axs[i, j].legend(fontsize=20)
        return axs


number_of_train_points = 1000
number_of_points = 100
constant_x = 0
constant_x_dot = 0
constant_theta = np.pi
constant_theta_dot = 0
delta_time = 0.2
sim_steps = 50

model = LinearModel(number_of_points=number_of_train_points, delta_time=delta_time, sim_steps=sim_steps)

linear_single_variable_scan = SingleVariableScan(num_search_points=number_of_points, x_constant=0.0, x_dot_constant=5,
                                            theta_constant=np.pi, theta_dot_constant=0, 
                                            model=model, delta_time=delta_time, sim_steps=sim_steps)
linear_single_variable_scan.compute_change()
oscillatory_single_variable_scan = SingleVariableScan(num_search_points=number_of_points, x_constant=0, x_dot_constant=0,
                                            theta_constant=np.pi, theta_dot_constant=0.5,
                                            model=model, delta_time=delta_time, sim_steps=sim_steps)
oscillatory_single_variable_scan.compute_change()
rotation_single_variable_scan = SingleVariableScan(num_search_points=number_of_points, x_constant=0, x_dot_constant=0,
                                            theta_constant=0, theta_dot_constant=15,
                                            model=model, delta_time=delta_time, sim_steps=sim_steps)
rotation_single_variable_scan.compute_change()

fig, axs = plt.subplots(4, 4, figsize=(20, 20), layout="constrained")
# linear_single_variable_scan.plot_change(fig, axs, " - Case: 1", linestyle1="dashed", linestyle2="dashed")
oscillatory_single_variable_scan.plot_change(fig, axs, " - Case: 2", linestyle1="solid", linestyle2="solid")
# rotation_single_variable_scan.plot_change(fig, axs, " - Case: 3", linestyle1="dashdot", linestyle2="dashdot")




# Have to choose the range of ticks for the y axis to be the same for each column
# For column 1, the range is from -2 to 2
# For column 2, the range is from -4 to 4
# For column 3, the range is from -3 to 3
# For column 4, the range is from -8 to 8

for ax in axs:
    ax[0].set_ylim([-2, 2])
    ax[1].set_ylim([-4, 4])
    ax[2].set_ylim([-4, 4])
    ax[3].set_ylim([-8, 8])
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=20, ncol=3)# bbox_to_anchor=(0, 10), fontsize=20)
fig.suptitle('Scan across each state variable - oscillation about stable equilibirum', fontsize=40,y=1.02)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(4, 4, figsize=(20, 20), layout="constrained")
rotation_single_variable_scan.plot_change(fig[], axs, linestyle1="solid", linestyle2="solid")


for ax in axs:
    ax[0].set_ylim([-2, 2])
    ax[1].set_ylim([-4, 4])
    ax[2].set_ylim([-4, 4])
    ax[3].set_ylim([-8, 8])
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=20, ncol=3)# bbox_to_anchor=(0, 10), fontsize=20)
fig.suptitle('Scan across each state variable - complete rotation of pendulum', fontsize=40,y=1.02)
plt.tight_layout()
plt.show()














# 1.4

def plot_real_and_model_over_time(X, C, no_its):
    # Initialise lists into which values to plot will go, with starting point added
    x_to_plot_actual, xdot_to_plot_actual, theta_to_plot_actual, thetadot_to_plot_actual = [X[0]], [X[1]], [X[2]], [X[3]]
    x_to_plot_pred, xdot_to_plot_pred, theta_to_plot_pred, thetadot_to_plot_pred = [X[0]], [X[1]], [X[2]], [X[3]]

    # Create array for time
    time_to_plot = np.arange(no_its)

    # Make instance of cartpole
    cp = CartPole()

    # Initilise initial values of actual X and predicted X = X
    actual_X, pred_X = X, X

    # Run simulation and prediction no_its times
    for i in range(no_its-1):
        # Update actual state and run one iteration to get new actual state
        cp.setState(actual_X)
        cp.performAction()
        new_actual_X = cp.getState()

        # Append new actual state variables to lists to plot for the actual case
        x_to_plot_actual.append(new_actual_X[0])
        xdot_to_plot_actual.append(new_actual_X[1])
        theta_to_plot_actual.append(new_actual_X[2])
        thetadot_to_plot_actual.append(new_actual_X[3])

        # Update actual X for next iteration
        actual_X = new_actual_X

        # Run simulation to get new predicted X
        Y = np.matmul(C, pred_X)
        new_pred_X = Y + pred_X

        # Remap angle so it stays within range -pi to pi
        new_pred_X[2] = _remap_angle(new_pred_X[2])

        # Append new predicted state variables to lists to plot for the predicted case
        x_to_plot_pred.append(new_pred_X[0])
        xdot_to_plot_pred.append(new_pred_X[1])
        theta_to_plot_pred.append(new_pred_X[2])
        thetadot_to_plot_pred.append(new_pred_X[3])

        # Update predicted X for next iteration
        pred_X = new_pred_X

    # Now plot graph of x
    plt.subplot(2, 2, 1)
    plt.plot(time_to_plot, x_to_plot_actual, label=r'Actual x')
    plt.plot(time_to_plot, x_to_plot_pred, label=r'Predicted x')
    plt.title(r'Time against x')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$x$')
    plt.legend()

    # Now plot graph of xdot
    plt.subplot(2, 2, 2)
    plt.plot(time_to_plot, xdot_to_plot_actual, label=r'Actual $\dot{x}$')
    plt.plot(time_to_plot, xdot_to_plot_pred, label=r'Predicted $\dot{x}$')
    plt.title(r'Time against $\dot{x}$')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$\dot{x}$')
    plt.legend()

    # Now plot graph of theta
    plt.subplot(2, 2, 3)
    plt.plot(time_to_plot, theta_to_plot_actual, label=r'Actual $\theta$')
    plt.plot(time_to_plot, theta_to_plot_pred, label=r'Predicted $\theta$')
    plt.title(r'Time against $\theta$')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$\theta$')
    plt.legend()

    # Now plot graph of thetadot
    plt.subplot(2, 2, 4)
    plt.plot(time_to_plot, thetadot_to_plot_actual, label=r'Actual $\dot{\theta}$')
    plt.plot(time_to_plot, thetadot_to_plot_pred, label=r'Predicted $\dot{\theta}$')
    plt.title(r'Time against $\dot{\theta}$')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$\dot{\theta}$')
    plt.legend()

    plt.suptitle('Graphs showing how the actual and predicted state variables change over time for oscillations')
    plt.tight_layout()
    plt.show()


# Get optimal matrix
no_datapoints = 5000
list_of_X_Y_tuples = make_XY_tuples(no_datapoints)
C = opt_coeff_matrix(list_of_X_Y_tuples)
X = np.array((0, 1, np.pi, 1))  # Oscillations
# X = np.array((0, 1, np.pi, 15))  # Complete circle
no_its = 50
plot_real_and_model_over_time(X, C, no_its)


def no_remap_plot_real_and_model_over_time(X, C, no_its):
    # Initialise lists into which values to plot will go, with starting point added
    x_to_plot_actual, xdot_to_plot_actual, theta_to_plot_actual, thetadot_to_plot_actual = [X[0]], [X[1]], [X[2]], [X[3]]
    x_to_plot_pred, xdot_to_plot_pred, theta_to_plot_pred, thetadot_to_plot_pred = [X[0]], [X[1]], [X[2]], [X[3]]

    # Create array for time
    time_to_plot = np.arange(no_its)

    # Make instance of cartpole
    cp = CartPole()

    # Initilise initial values of actual X and predicted X = X
    actual_X, pred_X = X, X

    # Run simulation and prediction no_its times
    for i in range(no_its-1):
        # Update actual state and run one iteration to get new actual state
        cp.setState(actual_X)
        cp.performAction()
        new_actual_X = cp.getState()

        # Append new actual state variables to lists to plot for the actual case
        x_to_plot_actual.append(new_actual_X[0])
        xdot_to_plot_actual.append(new_actual_X[1])
        theta_to_plot_actual.append(new_actual_X[2])
        thetadot_to_plot_actual.append(new_actual_X[3])

        # Update actual X for next iteration
        actual_X = new_actual_X

        # Run simulation to get new predicted X
        Y = np.matmul(C, pred_X)
        new_pred_X = Y + pred_X

        # Append new predicted state variables to lists to plot for the predicted case
        x_to_plot_pred.append(new_pred_X[0])
        xdot_to_plot_pred.append(new_pred_X[1])
        theta_to_plot_pred.append(new_pred_X[2])
        thetadot_to_plot_pred.append(new_pred_X[3])

        # Update predicted X for next iteration
        pred_X = new_pred_X

    # Now plot graph of x
    plt.subplot(2, 2, 1)
    plt.plot(time_to_plot, x_to_plot_actual, label=r'Actual x')
    plt.plot(time_to_plot, x_to_plot_pred, label=r'Predicted x')
    plt.title(r'Time against x')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$x$')
    plt.legend()

    # Now plot graph of xdot
    plt.subplot(2, 2, 2)
    plt.plot(time_to_plot, xdot_to_plot_actual, label=r'Actual $\dot{x}$')
    plt.plot(time_to_plot, xdot_to_plot_pred, label=r'Predicted $\dot{x}$')
    plt.title(r'Time against $\dot{x}$')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$\dot{x}$')
    plt.legend()
    
    # Now plot graph of theta
    plt.subplot(2, 2, 3)
    plt.plot(time_to_plot, theta_to_plot_actual, label=r'Actual $\theta$')
    plt.plot(time_to_plot, theta_to_plot_pred, label=r'Predicted $\theta$')
    plt.title(r'Time against $\theta$')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$\theta$')
    plt.legend()

    # Now plot graph of thetadot
    plt.subplot(2, 2, 4)
    plt.plot(time_to_plot, thetadot_to_plot_actual, label=r'Actual $\dot{\theta}$')
    plt.plot(time_to_plot, thetadot_to_plot_pred, label=r'Predicted $\dot{\theta}$')
    plt.title(r'Time against $\dot{\theta}$')
    plt.xlabel(r'Time (T)')
    plt.ylabel(r'$\dot{\theta}$')
    plt.legend()

    plt.suptitle('Graphs showing how the actual and predicted state variables change over time for oscillations')
    plt.tight_layout()
    plt.show()
    
no_remap_plot_real_and_model_over_time(X, C, no_its)



from CartPole import remap_angle

class CartPoleControl:
    def __init__(self, initial_state, force, time_steps, remap_angle=False, delta_time=0.2, sim_steps=50):
        self.initial_state = initial_state
        self.force = force
        self.time_steps = time_steps
        self.remap_angle = remap_angle
        self.t = np.arange(0, time_steps, 1)
        self.x = np.zeros(time_steps)
        self.x_dot = np.zeros(time_steps)
        self.theta = np.zeros(time_steps)
        self.theta_dot = np.zeros(time_steps)
        self.x[0], self.x_dot[0], self.theta[0], self.theta_dot[0] = initial_state
        self.time_multiplier = delta_time * sim_steps
        self.remapped = np.zeros(time_steps)
        self.delta_time = delta_time
        self.sim_steps = sim_steps

    def do_simulation(self):
        cartpole = CartPole()
        cartpole.setState(self.initial_state)
        cartpole.sim_steps = self.sim_steps
        cartpole.delta_time = self.delta_time
        for i in range(1, self.time_steps):
            cartpole.performAction(self.force(cartpole.getState))
            self.x[i], self.x_dot[i], self.theta[i], self.theta_dot[i] = cartpole.getState()
            if self.remap_angle:
                remapped = remap_angle(self.theta[i])
                self.theta[i] = remapped

    def plot_quantity(self, quantity, y_label, plot_label, ax=None, with_remap=False, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        time = self.t * self.time_multiplier
        if with_remap:
            # If this happens then instead of joining discontinuous sections of the angle theta, we want to not join them
            # This is because the angle theta is not continuous when remapped
            # Lets iterate through the array and find the discontinuities
            pass
        ax.plot(time, quantity, label=label, linewidth=5)
        ax.set_xlabel('t (s)', fontsize=20)
        ax.set_ylabel(plot_label, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.legend()
        return ax
    
    def internal_dictionary(self):
        dictionary= {'x': [self.x, 'x', 'x (m)'], 
                     'x_dot': [self.x_dot, '$\\dot{x}$', '$\\dot{x}$ (m/s)'],
                     'theta': [self.theta, '$\\theta$','$\\theta$ (rad)'],
                     'theta_dot': [self.theta_dot, '$\\dot{\\theta}$', '$\\dot{\\theta}$ (rad/s)']}
        return dictionary
    
    def plot(self, quantity, ax=None):
        dictionary = self.internal_dictionary()
        values, plot_label, y_label = dictionary[quantity]
        with_remap = (self.remap_angle) and (quantity == 'theta')
        return self.plot_quantity(values, y_label, plot_label, ax, with_remap)

    def plot_phase_portrait(self, quantity_1, quantity_2, xlabel, ylabel, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(quantity_1, quantity_2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    
    def phase_portrait(self, x, y, ax=None):
        dictionary = self.internal_dictionary()
        x_values, _, x_label = dictionary[x]
        y_values, _, y_label = dictionary[y]
        if self.remap_angle:
            if x == 'theta':
                x_values = remap_angle_array(x_values)
            elif y == 'theta':
                y_values = remap_angle_array(y_values)
        return self.plot_phase_portrait(x_values, y_values, x_label, y_label, ax)
    
    def plot_four_quantities(self, axs, label=None):
        dictionary = self.internal_dictionary()
        for ax, quantity in zip(axs, dictionary.keys()):
            ax = self.plot_quantity(*dictionary[quantity], ax, label=label)
        return axs

number_of_train_points = 2500
delta_time = 0.02
sim_steps = 50
    
random_model = LinearModel(number_of_points=number_of_train_points, delta_time=delta_time, sim_steps=sim_steps, remap_angle=False)

number_of_points= 300

def zero_force(state):
    return 0
                                                            
oscillatory_cartpole_initial_state = np.array([0, 0, np.pi, 15])

oscillatory_cartpole_iterative_model = LinearModel(number_of_points=number_of_points, # number of points to use for training is the same as that used for testing to try to see how much the model overfits,
                                                delta_time=delta_time,
                                                sim_steps=sim_steps,
                                                remap_angle=False,
                                                start_state=oscillatory_cartpole_initial_state,
                                                generate_next_state_randomly=False)

oscillating_cartpole_pole_control = CartPoleControl(initial_state=oscillatory_cartpole_initial_state, 
                                                    force=zero_force, 
                                                    remap_angle=True,
                                                    time_steps=number_of_points, 
                                                    delta_time=delta_time, 
                                                    sim_steps=sim_steps)
oscillating_cartpole_pole_control.do_simulation()

past_state, change = oscillatory_cartpole_iterative_model.get_data_itertatively(oscillatory_cartpole_initial_state)
# Along with visting the actual state space along this trajectory, I also want to visit all the nearby states
# I will do this by duplicating past state many times and adding a small amount of noise to each one
repeated_past_state = np.repeat(past_state, 100, axis=0)

noise_mean = 0
noise_std = 0.001
noise = np.random.normal(noise_mean, noise_std, repeated_past_state.shape)
new_past_state = repeated_past_state + noise
new_change = np.zeros(new_past_state.shape)


cartpole = CartPole()
cartpole.sim_steps = sim_steps
cartpole.delta_time = delta_time
for i in range(new_past_state.shape[0]):
    cartpole.setState(new_past_state[i])
    cartpole.performAction(0)
    new_change[i] = cartpole.getState() - new_past_state[i]
new_model = LinearModel()
new_model.train_model(new_past_state, new_change)

oscillating_cartpole_states = np.zeros((number_of_points, 4))
oscillating_cartpole_iterative_control = np.zeros((number_of_points, 4))
new_control = np.zeros((number_of_points, 4))

oscillating_cartpole_states[0] = oscillatory_cartpole_initial_state
oscillating_cartpole_iterative_control[0] = oscillatory_cartpole_initial_state
new_control[0] = oscillatory_cartpole_initial_state
for i in range(1, number_of_points):
    oscillating_cartpole_states[i] = random_model.predict_state(oscillating_cartpole_states[i-1], remap_theta=True)
    oscillating_cartpole_iterative_control[i] = oscillatory_cartpole_iterative_model.predict_state(oscillating_cartpole_iterative_control[i-1], 
                                                                                                    remap_theta=True)   
    new_control[i] = new_model.predict_state(new_control[i-1], remap_theta=True)

predicted_change = oscillatory_cartpole_iterative_model.predict_change(past_state)
new_predicted_change = new_model.predict_change(past_state)
random_predicted_change = random_model.predict_change(past_state)

fig, axs = plt.subplots(4, 1, figsize=(10, 8))
oscillating_cartpole_pole_control.plot_four_quantities(axs, label='Actual')
time = oscillating_cartpole_pole_control.t * oscillating_cartpole_pole_control.time_multiplier
for i in range(4):
    axs[i].plot(time, oscillating_cartpole_states[:, i], label='Model 1', linewidth=5, linestyle='solid')
    handles, labels = axs[i].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),  ncol=4, fontsize=20)
fig.suptitle("Linear Model for predicting", fontsize=20)
plt.tight_layout()