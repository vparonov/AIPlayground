import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import time

class AIMagicDarts:
    def __init__(self, onnx_file, min_data_points, max_data_points, input_features = 7):
        self.session = onnxruntime.InferenceSession(onnx_file)
        self.min_data_points = min_data_points
        self.max_data_points = max_data_points
        self.input_features = input_features
        self.reset()
        
    def reset(self):
        self.buffer = np.zeros(shape = (1, self.max_data_points, self.input_features), dtype =np.float32)
        #all initial points are invalid
        self.buffer[0, :, -1] = 1
  
        self.counter = 0 
        self.valid_points_counter = 0
 
    
    #input_data is array_like [x, y, z, vx, vy, vz] or None
    def tick(self, input_data=None):
 
        t_hit, x, z = None, None, None
        
        # if the buffer is full 
        # we drop the first point and keep the counter(pointer)
        # value 
        if self.counter == self.max_data_points:
            # if the first point in the buffer is valid
            # the valid_points_counter should be decreased
            # since this point will be dropped 
            if self.buffer[0, 0, -1] == 0:
                #print("decrease valid points counter from ",self.valid_points_counter )
                self.valid_points_counter -= 1
            
            #drop the first point from the buffer 
            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.counter = self.max_data_points - 1
 
        if input_data is None:
            self.buffer[0, self.counter, :-1] = 0
            self.buffer[0, self.counter, -1]  = 1 
        else:
            self.buffer[0, self.counter, :-1] = input_data
            self.buffer[0, self.counter, -1]  = 0
            #print("increase valid points counter from ",self.valid_points_counter )
            self.valid_points_counter += 1
            
            # if the minimum valid measurements are in the buffer
            # the model is asked for prediction
            if self.valid_points_counter >= self.min_data_points:
                #debug_valid_points = self.get_valid_points()
                #print("debug valid points = ", debug_valid_points)
                inputs = {self.session.get_inputs()[0].name: self.buffer}
                outputs = self.session.run(None, inputs)[0]
                t_hit = outputs[0][0]
                x = outputs[0][1]
                z = outputs[0][2]
            
        if self.counter < self.max_data_points:
            self.counter += 1
            
        return t_hit, x, z,self.valid_points_counter 
        
    def get_valid_points(self):
        return self.max_data_points - self.buffer[0,:,-1].sum()

plt.rcParams["figure.figsize"] = (22,10)

def plot_traj_zy(traj_points):
    traj_points[traj_points[:,-1] == 1, :-1] = float('NaN')
    plt.plot(-traj_points[:, 2], traj_points[:, 3], '--', marker= 'x')
    traj_points[traj_points[:,-1] == 1, :-1] = 0

################################
# input parameters 
#onnx_model_file = 'magic_darts_the_nextG_10to20_cuda.onnx'
#onnx_model_file = 'magic_darts_the_nextG_10to20_trained_with_noise.onnx'
#min_data_points = 10
#max_data_points = 20
#
onnx_model_file = 'magic_darts_the_nextG_25to30_points_trained_with_noise.onnx'
min_data_points = 25
max_data_points = 30

#test_inputs  = 'test1_81_trajectories_inputs.npy'
#test_outputs = 'test1_81_trajectories_outputs.npy'
#test_inputs  = 'test2_2000_trajectories_inputs.npy'
#test_outputs = 'test2_2000_trajectories_outputs.npy'
test_inputs  = 'test3_2000_with_noise_trajectories_inputs.npy'
test_outputs = 'test3_2000_with_noise_trajectories_outputs.npy'

noiseless_test_inputs  = 'test3_2000_no_noise_trajectories_inputs.npy'
noiseless_test_outputs = 'test3_2000_no_noise_trajectories_outputs.npy'

#the probability a certain point to be dropped (0..1)
drop_point_prob = 0
time_to_impact_th1 = 150e-3
time_to_impact_th2 = 50e-3

stats_first_detection_distance = []
stats_th1_detection_distance = []
stats_th2_detection_distance = []
stats_last_detection_distance = []

##################################
ai_darts = AIMagicDarts(onnx_model_file, min_data_points, max_data_points)

inputs = np.load(test_inputs)
outputs = np.load(test_outputs)


noiseless_inputs = np.load(noiseless_test_inputs)
noiseless_outputs = np.load(noiseless_test_outputs)

times = []

show_plots = False 
print_outputs = False 

N = 2000

for traj_ix in range(N):
    th1_line_shown = False 
    th2_line_shown = False 
    first_point_shown = False

    traj_points = inputs[traj_ix]
    hit_point = outputs[traj_ix]
    
    noiseless_traj_points = noiseless_inputs[traj_ix]
    
    hp_x = hit_point[1]
    hp_z = hit_point[2]

    #print(traj_points)
    #print("----------------")
    #print(traj_points[traj_points[:,-1] == 0, :])
    if show_plots:
        plt.scatter(0, hp_z, marker='x', c='g', s=150)
        plt.annotate(f"true hit point", (0, hp_z))
        plot_traj_zy(traj_points)
        plot_traj_zy(noiseless_traj_points)
        
            
    ai_darts.reset()
    
    #the starting point in simulation mode could be different from 0
    #(e.g. randomly chosen between 0 and end_point)
    start_point = 0
    end_point = traj_points.shape[0]

    stats = np.empty((start_point))

    for j in range(start_point, end_point):
        print_outputs and print('--------------------------------')
        
        current_point = traj_points[j, :]
        
        is_invalid_point = current_point[-1] == 1
        # tick without data is done in two cases:
        #   1) when the trajectory point is marked as invalid
        #   2) when simulated 'drop' of measurement has occured 
        # NB! j == 0 is the first point 
        # and by convention the first point is always valid
        if  is_invalid_point or (j > 0 and np.random.rand() < drop_point_prob):
            start = time.time()
            _, _, _, vp = ai_darts.tick()
            end = time.time()
            times.append(end - start)
            print_outputs and print(f'{j} just tick; valid points = {vp},  elapsed = {1000 * (end - start):.3f} ms')
            if show_plots and not is_invalid_point:
                plt.scatter(-current_point[2], current_point[3], marker='x', c='r')
        else:
            hp_time = hit_point[0] - current_point[0]

            start = time.time()
            hp_time_hat, hp_x_hat, hp_z_hat, vp = ai_darts.tick(current_point[1:-1])
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            
            if hp_time_hat is None:
                print_outputs and print(f'{j} not enough points; valid points = {vp},  elapsed = {1000 * (end - start):.3f} ms')
                show_plots and plt.scatter(-current_point[2], current_point[3], marker='.', c='b')
            else:
                real_distance = np.sqrt((hp_x_hat - hp_x)**2 + (hp_z_hat - hp_z)**2)
                real_hit_time_difference = np.sqrt((hp_time - hp_time_hat)**2)
                show_plots and plt.scatter(-current_point[2], current_point[3], marker='o', c='g')
        
                last_non_zero_y = current_point[2]
                last_non_zero_z = current_point[3]
                
                if not first_point_shown:
                    stats_first_detection_distance.append(real_distance)
                    if show_plots: 
                        plt.axvline(x=-current_point[2], linestyle='--')
                        plt.annotate(f'(f) ttm = {hp_time*1000:.0f}ms, predicted = {hp_time_hat*1000:.0f}ms, delta = {(hp_time-hp_time_hat)*1000:.0f}ms\nerror = {real_distance:.5f}m', 
                                    (-current_point[2]+0.003, current_point[3]+0.006))
                        plt.scatter(0, hp_z_hat, marker='o', c='r') 
                        plt.annotate(f"@first", (0, hp_z_hat))
                    first_point_shown = True 

                if not th1_line_shown and hp_time_hat <= time_to_impact_th1:
                    stats_th1_detection_distance.append(real_distance)
                    if show_plots: 
                        plt.axvline(x=-current_point[2], linestyle='--')
                        plt.annotate(f'(t1) tti = {hp_time*1000:.0f}ms, predicted = {hp_time_hat*1000:.0f}ms, delta = {(hp_time-hp_time_hat)*1000:.0f}ms\nerror = {real_distance:.5f}m', 
                                    (-current_point[2]+0.003, current_point[3]+0.006))
                        plt.scatter(0, hp_z_hat, marker='o', c='r') 
                        plt.annotate(f"@th1", (0, hp_z_hat))
                    th1_line_shown = True 

                if not th2_line_shown and hp_time_hat <= time_to_impact_th2:
                    stats_th2_detection_distance.append(real_distance)
                    if show_plots:  
                        plt.axvline(x=-current_point[2], linestyle='--')
                        plt.annotate(f'(t2) tti = {hp_time*1000:.0f}ms, predicted = {hp_time_hat*1000:.0f}ms, delta = {(hp_time-hp_time_hat)*1000:.0f}ms\nerror = {real_distance:.5f}m', 
                                    (-current_point[2]+0.003, current_point[3]+0.006))
                        plt.scatter(0, hp_z_hat, marker='o', c='c')                
                        plt.annotate(f"@th2", (0, hp_z_hat))
                    th2_line_shown = True 
                    
                if print_outputs:
                    print(f"{j} vp = {vp} (actual)    elapsed = {1000 * (end - start):.3f}; time to impact = {hp_time*1000:.3f} ms @({hp_x:.3f}, {hp_z:.3f}) m")
                    print(f'{j} vp = {vp} (predicted) elapsed = {1000 * (end - start):.3f}; time to impact = {hp_time_hat*1000:.3f} ms @({hp_x_hat:.3f}, {hp_z_hat:.3f}), distance = {real_distance:.5f}m, rms time = {1000 * real_hit_time_difference:.3f} ms')
    if show_plots and last_non_zero_y > 0 and  last_non_zero_z > 0:
        plt.axvline(x=-last_non_zero_y, linestyle='--')
        try:
            plt.annotate(f'(l) tti = {hp_time*1000:.0f}ms, predicted = {hp_time_hat*1000:.0f}ms, delta = {(hp_time-hp_time_hat)*1000:.0f}ms\nerror = {real_distance:.5f}m', 
                    (-last_non_zero_y+0.003, last_non_zero_z+0.006))
        except:
            pass
        plt.annotate(f"@last", (0, hp_z_hat))

    if show_plots:
        plt.title(f'file: {test_inputs}, trajectory = {traj_ix}')
        plt.show()

plt.boxplot([stats_first_detection_distance,stats_th1_detection_distance,stats_th2_detection_distance], 
           labels = ['first detection', 'est. TTI < 150ms', 'est. TTI < 50ms'])
plt.ylabel("distance [m]")
plt.show()
#print(stats_th2_detection_distance)
tt =np.array(times)*1000
plt.hist(tt)
plt.show()
print(tt.min(), tt.max(), tt.mean())