import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Adaptar a las rutas de los archivos
eventos = pd.read_json("../../data/simulation.jsonlines", lines=True)
planes = pd.read_json("../../data/plans.jsonlines", lines=True)

camiones = []
for simId in planes.simulationId.unique():
    for truck in planes[planes.simulationId == simId].trucks.values[0]:
        camiones.append(pd.DataFrame(truck["route"]).assign(simulationId=simId, truckId=truck["truck_id"]))
camiones = pd.concat(camiones)

tiempos_plan = camiones.sort_values(["simulationId","truckId"]).assign(duration=lambda x: x["duration"]*1000).groupby(["simulationId","truckId"]).duration.agg(list).reset_index()
tiempos_plan.rename(columns={"duration":"tiempo_plan"}, inplace=True)

eventos = eventos.sort_values(["simulationId", "truckId", "eventTime"])
eventos["prev_event"] = eventos.groupby(["truckId", "simulationId"])["eventType"].shift(1)
eventos["prev_time"] = eventos.groupby(["truckId", "simulationId"])["eventTime"].shift(1)
eventos["delta"] = eventos.eventTime - eventos.prev_time
tiempos_sim = eventos[eventos.eventType.isin(["Truck arrived", "Truck ended route"])].sort_values(["simulationId","truckId", "eventTime"]).groupby(["simulationId","truckId"]).delta.agg(list).reset_index()
tiempos_sim.rename(columns={"delta":"tiempo_sim"}, inplace=True)

retrasos = tiempos_sim.merge(tiempos_plan, on=["simulationId","truckId"]).dropna().reset_index(drop=True)

arr = np.array(retrasos.apply(lambda x: list(zip(x.tiempo_plan, x.tiempo_sim)), axis=1).explode())
arr = np.array(arr.tolist())
x = arr[:,0].reshape(-1,1) / 1000
y = arr[:,1].reshape(-1,1) / 1000


# Entrenar modelo
travelModel = LinearRegression()
travelModel.fit(x, y)

# Guardar modelo
with open('../../models/travelModel.pkl', 'wb') as f:
    pickle.dump(travelModel, f)

tiemposEntrega = eventos[eventos.eventType=="Truck ended delivering"][["truckId", "delta"]]

# Label encoding
le = LabelEncoder()
tiemposEntrega["truckId"] = le.fit_transform(tiemposEntrega["truckId"])

# Entrenar modelo
deliveryModel = LinearRegression()
deliveryModel.fit(tiemposEntrega["truckId"].values.reshape(-1,1), tiemposEntrega["delta"].values.reshape(-1,1))

# Guardar modelo y label encoder
with open('../../models/deliveryModel.pkl', 'wb') as f:
    pickle.dump(deliveryModel, f)
    
with open('../../models/le.pkl', 'wb') as f:
    pickle.dump(le, f)