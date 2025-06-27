from sklearn.linear_model import LinearRegression                   # pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator 
import numpy as np                                                  # pip install numpy
import pandas as pd                                                 # pip install pandas
import matplotlib.pyplot as plt                                     # pip install matplotlib
                                                                    # pip install scikit-learn matplotlib numpy pandas

data_frame = pd.read_csv("epl_player_stats_24_25.csv", sep=";")
print(type(data_frame))

                    #Recoleccion y preparacion de datos

# Exploración de datos
print("Primeras filas del DataFrame:")
print(data_frame.head())

print("\nResumen estadistico")
print(data_frame.describe())

print("\nValores nulos por columna:")
print(data_frame.isnull().sum())

                    # Limpieza de datos
# 1. Eliminar filas con valores nulos
data_frame = data_frame.dropna()

# 2. Eliminar duplicados si existen
data_frame = data_frame.drop_duplicates()

                # Transformación de Datos

# 1. Crear una nueva columna: Goles por Partido
data_frame["Goles_por_partido"] = data_frame["Goles"] / data_frame["Apariciones"]
print("\nNuevo DataFrame)")
print(data_frame) 


# 2. Normalizacion de datos: Normalizar la columna de disparos al arco
if "Disparos_al_arco" in data_frame.columns:
    max_disparos = data_frame["Disparos_al_arco"].max()
    data_frame["Disparos_al_arco"] = data_frame["Disparos_al_arco"] / max_disparos

# 3. Filtrar jugadores con más de 5 goles
if "Goles" in data_frame.columns:
    data_frame = data_frame[data_frame["Goles"] > 5]

# 4. Agrupar por equipo y calcular la media de goles
if "Club" in data_frame.columns and "Goles" in data_frame.columns:
    equipo_goles = data_frame.groupby("Club")["Goles"].mean().reset_index()
    equipo_goles.columns = ["Club", "Media_Goles"]
    data_frame = data_frame.merge(equipo_goles, on="Club", how="left")

print("\nDataFrame después de la transformación:")
print(data_frame.head())


                     #Analisis de Datos


            # Análisis Exploratorio de Datos (EDA)

# 1. Estadísticas descriptivas generales
print("\nEstadísticas descriptivas del DataFrame transformado:")
print(data_frame.describe())

# 2. Histograma de la distribución de goles
if "Goles" in data_frame.columns:
    plt.figure(figsize=(8,4))
    plt.hist(data_frame["Goles"], bins=15, color='skyblue', edgecolor='black')
    plt.title("Distribución de Goles")
    plt.xlabel("Goles")
    plt.ylabel("Cantidad de Jugadores")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) # Asegura que el eje y tenga valores enteros
    plt.show()


# 3. Tendencia preliminar: Goles por equipo
if "Club" in data_frame.columns and "Goles" in data_frame.columns:
    plt.figure(figsize=(10,5))

    # Agrupar por club y sumar los goles 
    equipos = data_frame.groupby("Club", as_index=False)["Goles"].sum().sort_values("Goles", ascending=False)

    # Gráfico de barras de goles por equipo
    plt.bar(equipos["Club"], equipos["Goles"], color="orange")
    plt.title("Goles por Equipo")
    plt.xlabel("Club")
    plt.ylabel("Total de Goles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

     # Analisis estadisticos 

# 1. Verificar si las columnas necesarias existen
if "Goles" in data_frame.columns and "Apariciones" in data_frame.columns:
    # Filtrar datos válidos
    data = data_frame[["Goles", "Apariciones"]].dropna()

    # Variables independientes (X) y dependientes (y)
    X = data["Apariciones"].values.reshape(-1, 1)
    y = data["Goles"].values

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nResultados de la Regresión Lineal:") 
    print(f"Coeficiente (pendiente): {model.coef_[0]}")
    print(f"Intercepto: {model.intercept_}")
    print(f"Error Cuadrático Medio (MSE): {mse}") 
    print(f"Coeficiente de Determinación (R^2): {r2}")

    # Interpretación
    if r2 > 0.5:
        print("\nExiste una relación moderada o fuerte entre las apariciones y los goles.")
    else:
        print("\nLa relación entre las apariciones y los goles es débil.")

# 2. Gráfico de dispersión con línea de regresión
plt.scatter(data["Apariciones"], data["Goles"], color="blue", label="Datos reales")
plt.plot(data["Apariciones"], model.predict(X), color="red", label="Línea de regresión")
plt.title("Relación entre Apariciones y Goles")
plt.xlabel("Apariciones")
plt.ylabel("Goles")
plt.legend()
plt.grid()
plt.show()
