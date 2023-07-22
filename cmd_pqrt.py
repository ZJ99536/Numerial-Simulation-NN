import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from tensorflow import keras

class DroneControlSim:
    def __init__(self):
        self.sim_time = 5.0
        self.sim_step = 0.03
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.drone_states[0, 0] = -2.0
        self.drone_states[0, 1] = 0.0
        self.drone_states[0, 2] = 1.0
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 1.0
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

    def drone_dynamics(self,T,M):
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]

        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        d_position = np.array([vx,vy,vz])
        d_velocity = np.array([.0,.0,-self.g]) + R_E_B.transpose()@np.array([.0,.0,T])
        d_angle = R_d_angle@np.array([p,q,r])
        # d_angle = np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        return dx 

    def run(self):
        waypoint0 = np.array([0.0, 1.0, 1.2])
        waypoint1 = np.array([2.0, 0.0, 1.0])
        model = keras.models.load_model('/home/zhoujin/learning/model/quad5_m5.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        # model = keras.models.load_model('/home/zhoujin/learning/model/quad3_t3.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        last_velocity = np.zeros(3)
        for self.pointer in range(self.drone_states.shape[0]-1):
            input = np.zeros(15)
            current_position = np.array([self.drone_states[self.pointer, 0], self.drone_states[self.pointer, 1], self.drone_states[self.pointer, 2]])
            current_velocity = np.array([self.drone_states[self.pointer, 3], self.drone_states[self.pointer, 4], self.drone_states[self.pointer, 5]])
            error0 = waypoint0 - current_position
            error1 = waypoint1 - current_position
            
            for i in range(3):
                input[i] = error0[i]
                input[i+3] = error1[i]
                input[i+6] = self.drone_states[self.pointer, i+3]
                input[i+9] = (current_velocity[i] - last_velocity[i]) / self.sim_step
                input[i+12] = self.drone_states[self.pointer, i+9]
            input[11] += 9.81
            last_velocity = current_velocity
            output = model(input.reshape(-1,15))

            self.time[self.pointer] = self.pointer * self.sim_step
            psi_cmd = 0.0
            
            x_cmd = ((waypoint0[0] - output[0, 0]) + (waypoint1[0] - output[0, 3])) / 2
            y_cmd = ((waypoint0[1] - output[0, 1]) + (waypoint1[1] - output[0, 4])) / 2
            z_cmd = ((waypoint0[2] - output[0, 2]) + (waypoint1[2] - output[0, 5])) / 2
        
            self.position_cmd[self.pointer] = [x_cmd, y_cmd, z_cmd]
            self.velocity_cmd[self.pointer] = [output[0, 6], output[0, 7], output[0, 8]]
            
            # self.attitude_cmd[self.pointer] = [output[0, 9], output[0, 10], output[0, 11]]

            #self.attitude_cmd[self.pointer] = [1,0,0]
            self.rate_cmd[self.pointer] = [output[0, 12], output[0, 13], output[0, 14]]
            thrust_cmd = 0
            for i in range(3):
                thrust_cmd += output[0, i+9] ** 2
            thrust_cmd = thrust_cmd ** 0.5
            # self.rate_cmd[self.pointer] = [1,0,0]
            self.attitude_cmd[self.pointer] = [output[0, 9], output[0, 10], thrust_cmd]
            # self.attitude_cmd[self.pointer] = [input[9], input[10], input[11]]

            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m
            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 0.1 
        kp_q = 0.1
        kp_r = 0.2
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])

   

    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()