import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

misDatos = datasets.load_digits()

print(f"Claves del dataset: {misDatos.keys()}")
print(misDatos["images"][1000])

plt.imshow(misDatos['images'][1000], cmap="gray")
plt.show()

print(misDatos['target'][1000])
print(misDatos['target_names'])
print(misDatos["DESCR"])


df1 = pd.DataFrame(data = misDatos['images'][0])
separador = pd.DataFrame(data=[[0,0,0,0,0,0,0,0]])
df2 = pd.DataFrame(data= misDatos['images'][1])

df1 = pd.concat([df1, separador, df2, separador], ignore_index=True)

for i in range(2,1797):
    df2 = pd.DataFrame(data=misDatos['images'][i])
    df1 = pd.concat([df1,df2, separador], ignore_index=True)

lista = []

for etiqueta in misDatos['target']:
    for _ in range(9):
        lista.append(etiqueta)


df1["Etiqueta"] = lista

df1.to_csv("digitos.csv", index= False)
#Primer paso