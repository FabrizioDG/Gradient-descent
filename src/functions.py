import numpy as np

def cost_function(X,y,w,b):
    """
    Evaluate the cost function 1/n sum (y - yhat)**2

    Inputs:
    X (ndarray (m,n)): Data, m training examples, n features
    y ((ndarray (m,)): Target, m examples 
    w (ndarray (n,)): weights, n features
    b (scalar) : bias
     
    Output:
    dj_dw (ndarray(,n)) : gradient w.r.t parameters w
    dj_db (scalar) : gradient w.r.t. parameter b
    """
    m, n = X.shape
    cost = 1/m * np.sum((np.dot(X, w) + b - y)**2)
    return cost

def gradient (X,y,w,b, clip_value = None):
    """
    Evaluate the gradient of the cost function w.r.t. the parameters w and b
    Inputs:
    X (ndarray (m,n)): Data, m training examples, n features
    y ((ndarray (m,)): Target, m examples 
    w (ndarray (n,)): weights, n features
    b (scalar) : bias
     
    Outputs:
    dj_dw (ndarray(,n)) : gradient w.r.t parameters w
    dj_db (scalar) : gradient w.r.t. parameter b
    """
    if type(w) != np.ndarray:
        w = np.array([w])
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    dj_dw = np.matmul((np.dot(X,w) + b-y),X)*2/m

    dj_db = np.sum((np.dot(X,w) + b-y))*2/m

    # Not optimized code
    #for i in range(m):
    #    dj_dw_i = (np.dot(X[i],w) + b - y[i])
    #    for j in range(n):
    #        dj_dw[j] = dj_dw[j] + dj_dw_i * X[i,j]
    #    dj_db+= (np.dot(X[i],w) + b - y[i])

    #dj_dw = 2/m * dj_dw 
    #dj_db = 2/m * dj_db
    
    #clipping gradients
    if clip_value!=None:
        if np.abs(dj_dw) > clip_value :
            dj_dw = dj_dw/np.abs(dj_dw) * clip_value
        if np.abs(dj_db) > clip_value :
            dj_db = dj_db/np.abs(dj_db) * clip_value
    return dj_dw, dj_db


def gradient_descent(X,y,w_in,b_in, alpha, tolerance = 1e-10, num_iter = 100000, clip_value=None):
    """
    Gradient descent to minimize the cost function
    
    Inputs:
    X (ndarray (m,n)): Data, m training examples, n features
    y ((ndarray (m,)): Target, m examples 
    w_in (ndarray (n,)): initial guess for weights, n features
    b_in (scalar) : initial guess for bias
    alpha (scalar) : learning rate
    tolerance (scalar) : tolerance for convergence
    num_iter (scalar) : maximum number of iterations
     
    Outputs:
    w (ndarray(,n)) : value of parameters w at the minimum of the cost function
    b (scalar) : value of paramer b at the minimum of the cost function
    cost_history (list) : history of cost function value at each iterations
    param_history (list) : history of parameters at each iterations
    """ 
    if type(w_in) != np.ndarray:
        w_in = np.array([w_in])
    m, n = X.shape
    w_old = w_in
    b_old = b_in
    cost_history = [cost_function(X,y,w_in,b_in)]
    param_history = [(np.array([w_in]), b_in)]
    i = 0
    while (i<num_iter):
        w = w_old - alpha*gradient(X,y,w_old,b_old, clip_value)[0]
        b = b_old - alpha*gradient(X,y,w_old,b_old, clip_value)[1]
        if (i<10000):
            cost_history.append(cost_function(X,y,w,b))
            param_history.append((w,b))
        if (all([np.abs(w[i]-w_old[i])<tolerance for i in range(len(w))]) and np.abs(b-b_old)<tolerance):
            break
        w_old = w
        b_old = b
        i+=1
    if i == num_iter :
        print(f"Didn't converge, tolerance reached: {alpha*gradient(X,y,w_old,b_old, clip_value)[0], alpha*gradient(X,y,w_old,b_old, clip_value)[1]}")
    else:
        print("Tolerance reached at iteration: ", i)

    return w,b, cost_history, param_history



def create_frame(t, df_cost_param, df):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))

    ax1.plot(df["x"], df_cost_param["w"].iloc[t]*df["x"] + df_cost_param["b"].iloc[t],
            color = 'blue', label = "gradient descent" )
    ax1.scatter(df["x"],df["y"], color = 'red', marker = 'o', label = "data" )
    ax1.set_xlim([0,10])
    ax1.set_ylim([0,7.5])

    ax1.set_xlabel('x', fontsize = 14)
    ax1.set_ylabel('y', fontsize = 14)
    ax1.set_title(f'Evolution of gradient descent solution at iteration {t}',
                fontsize=14)
    ax1.legend()
    ax1.text(0.2, 5.5, r'$\frac{1}{n} \sum_0^n (y - (mx+b))^2$' + " = %0.3f" %df_cost_param['cost'].iloc[t],
            fontsize = 13) 

    ax2.plot(list(range(t)), df_cost_param["cost"].iloc[0:t], color = 'green')
    ax2.set_title(f'Evolution of the cost function at iteration {t}', fontsize=14)

    ax2.set_xlabel('iteration', fontsize = 14)
    ax2.set_ylabel('Cost function', fontsize = 14)

    plt.savefig(f'../img/gradientDescent_{t}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()

def create_gif(df_cost_param, df, num_frames, fps):
    import imageio
    for t in range(num_frames):
        create_frame(t, df_cost_param, df)

    frames = []
    for t in range(num_frames):
        image = imageio.v2.imread(f'../img/gradientDescent_{t}.png')
        frames.append(image)
    imageio.mimsave('../img/gradientDescent.gif', # output gif
                frames,          # array of input frames
                fps = fps)         # optional: frames per second