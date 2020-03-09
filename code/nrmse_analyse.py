import matplotlib.pyplot as plt

nrmse = pd.read_csv('../../tmp/nrmse_20200225.csv')
target = pd.read_csv('../../tmp/target_region_0_4_20200225.csv')



plt.figure()
plt.plot([target.values[i,:].mean() for i in range(target.shape[0])], 'r',label="target")
plt.plot(nrmse.values, 'b',label = 'nrmse')
plt.legend()
for x,values in enumerate(nrmse.values):
    if(values > 0.8):
        plt.plot([x for i in range(10)],[ i for i in range(10)],alpha=0.4,color ='orange')
plt.show()