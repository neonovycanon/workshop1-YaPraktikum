import pandas as pd
import numpy as np
import os

#1 Получение пути к данным
def path_check(path_name):
  '''Проверка существования пути к файлу'''
  if os.path.exists(os.getcwd()+path_name):
      print('OK - Path exists')
      return os.getcwd()+path_name
  else:
      raise EnvironmentError('Нет пути к данным') 

#2 Оценка количества пропущенных значений
def unn_info_pct(data):
  '''Вывод информации о пропущенных значениях
  Аргумент
    data: pd.DataFrame
  Возвращаемые данные:
    pd.DataFrame с количественным и %-ным представлением о пропущенных значениях'''
  return pd.DataFrame([data.isna().sum(), data.isna().sum() / len(data) * 100]).T
