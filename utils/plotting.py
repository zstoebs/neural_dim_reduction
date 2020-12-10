import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper', font_scale=2.0)
sns.set_style('ticks')

### plotting helpers

def meta(title,xlab,ylab,ymin=-4,ymax=3,stride=1):
    sns.despine()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.yticks(range(int(ymin),int(ymax),stride))
    plt.ylim([ymin-0.5, ymax+0.5])
    plt.xticks(np.linspace(-0.4,.8,4))
    plt.xlim([-.5, .9])
    plt.title(title)

def plot_by_cell_types(X,G,features,title,savename="waveform_by_cell_type"):

    # creating a long-form dataframe for seaborn plotting
    points = []
    r, c = X.shape
    for i in range(r):
        for j in range(c):
            points += [(features[j],X[i,j],G[i])]

    norm_data = pd.DataFrame(points,columns=['timepoint','signal','cell'])

    sns.relplot(data=norm_data,x='timepoint',y='signal',hue='cell',kind='line')
    meta(title,'Time (ms)','z-score',X.min(),X.max())
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    fig.savefig('imgs/' + savename + '.png')

# modified 3D plotting code from: https://stackabuse.com/seaborn-scatter-plot-tutorial-and-examples/
def plot_3d(dim_red_data,labels,title,savename="dim_red_data"):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    colors = ['red','green','blue','purple','yellow','lime','orange','cyan','salmon','navy','black','gray','magenta']
    for i in range(len(labels)):
        x_dims = dim_red_data.Comp1[dim_red_data.y==i]
        y_dims = dim_red_data.Comp2[dim_red_data.y==i]
        z_dims = dim_red_data.Comp3[dim_red_data.y==i]

        ax.scatter(x_dims, y_dims, z_dims,color=colors[i],label=labels[i])

    plt.legend(loc="upper left")
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('imgs/' + savename + '.png')
    plt.show()

# modified functions from: https://jmausolf.github.io/code/pca_in_python/
def scree_plot(exp_var,savename="scree_plot"):

    plt.plot(range(1,len(exp_var)+1),exp_var)
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot of PCA')
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    fig.savefig('imgs/' + savename + '.png')
    plt.show()

def var_explained(exp_var_ratio,savename="cumulative_exp_var"):

    plt.plot(range(1,len(exp_var_ratio)+1),np.cumsum(exp_var_ratio))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained variance')
    plt.title('Cumulative Explained Variance')
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    fig.savefig('imgs/' + savename + '.png')
    plt.show()
