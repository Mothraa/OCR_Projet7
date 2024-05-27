import json
import time
import sys
import tracemalloc
import gc
from functools import wraps
from collections import Counter
# import numpy as np

"""Traitement préalable : 
    - conversion des données d'origine en json (.data//actions_source.json)
    - conversion de la 3eme colonne (bénéfice à 2 ans en %)
    par la valeur brute du bénéfice en €. ATTENTION : arrondi à 2 digits
    => .data//fichier actions_tuples_benefice.json

    solution du problème du sac à dos (KP ou Knapsack)
    Algorithme par Programmation Dynamique

    pseudo-code :
        pour c de 0 à W faire
            T[0,c] := 0
        fin pour

        pour i de 1 à n faire
            pour c de 0 à W faire
                si c>=w[i] alors
                    T[i,c] := max(T[i-1,c], T[i-1, c-w[i]] + p[i])
                sinon
                    T[i,c] := T[i-1,c]
                fin si
            fin pour
        fin pour
        source : https://fr.wikipedia.org/wiki/Probl%C3%A8me_du_sac_%C3%A0_dos

        Il a deux avantages :
            rapide si les poids sont entiers (et pas trop grande dispersion des valeurs ?), et la capacité du sac modérée.
            pas besoin de trier les variables.

        et un inconvénient :
            gourmand en mémoire (donc pas de résolution de problèmes de grande taille).

        Il est à noter que cet algorithme ne s’exécute pas en temps polynomial par rapport à la taille de l'entrée.
        En effet la complexité étant proportionnelle à la capacité du sac W, elle est exponentielle par rapport à son codage.
        Si les poids des objets sont décimaux, cela oblige à multiplier les poids des objets et la capacité du sac afin de les rendre entiers.
        Cette opération peut alors rendre l'algorithme très lent.

analyse de la conso mémoire : https://www.ukonline.be/cours/python/opti/chapitre3-5
"""

MAX_AMOUNT = 500  # montant max en euros par clientsv
MULTIPLIER_FACTOR = 100  # for removing decimal values => transform to int
TEXT_SEPARATOR = "-" * 20
NB_MEM_STATS = 3
FILE_PATH_INPUT = './/data//actions_tuples_benefice_dataset2.json'
FILE_PATH_OUTPUT = './/output//result.txt'
# file_path = './/data//actions_tuples_benefice_dataset2.json'
# file_path = './/data//actions_tuples_benefice.json'


class DataCleaning():
    def __init__(self) -> None:
        pass




with open(FILE_PATH_INPUT, 'r') as file:
    data = json.load(file)


def redir_stdout_to_file(file_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(file_path, 'w') as f:
                sys.stdout = f
                result = func(*args, **kwargs)
                sys.stdout = original_stdout
            return result
        return wrapper
    return decorator


def measure_memory(func):
    @wraps(func)  # permet de conserver les attributs spéciaux (module de la lib standard)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        gc.collect()  # on lance un garbage collector pour libérer la mémoire qui peut l'etre
        gc.disable()  # on desactive le GC pour éviter qu'il fonctionne pendant la mesure de la mémoire
        mem_snap_before = tracemalloc.take_snapshot()
        result = func(*args, **kwargs)
        mem_snap_after = tracemalloc.take_snapshot()
        gc.enable()  # Réactive le garbage collector
        top_stats = mem_snap_after.compare_to(mem_snap_before, 'lineno')
        print(TEXT_SEPARATOR)
        # on affiche les 3 consommations de mémoire les plus significatives
        for stat in top_stats[:NB_MEM_STATS]:
            print(stat)
        print(TEXT_SEPARATOR)
        return result
    return wrapper


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(TEXT_SEPARATOR)
        print(f"temps d'éxecution : {execution_time:.4f} secondes")
        return result
    return wrapper


# @measure_memory
def initialize_matrix(amount, selected_actions):
    # on initialise la matrice (2 listes imbriquées) à 0 (montant+1 * nombre d'actions+1)
    return [[0 for _ in range(amount + 1)] for _ in range(len(selected_actions) + 1)]


# @measure_memory
# @measure_time
def calculate_optimized_values(matrix, amount, selected_actions):
    # on parcours chaque action
    for i in range(1, len(selected_actions) + 1):
        # pour chaque element on parcours la "capacité du sac" (montant) > les colonnes de la matrice
        for j in range(1, amount + 1):
            # Valeur optimisée sans inclure l'action actuelle
            previous_optimized_value = matrix[i-1][j]
            # si cout de l'element inférieur au montant(colonne), on peut mettre l'element dedans
            if selected_actions[i-1][1] <= j: 
                # bénéfice de l'action actuelle
                current_benefit = selected_actions[i-1][2]
                # valeur optimisée pour le montant restant
                remaining_optimized_value = matrix[i-1][j - selected_actions[i-1][1]]
                # valeur optimisée si on inclus l'action actuelle
                new_optimized_value = current_benefit + remaining_optimized_value
                # on prend la valeur max entre la valeur qui inclus l'action en cours et celle qui ne l'inclus pas
                current_value = max(new_optimized_value, previous_optimized_value)
            else:  # si cout supérieur au montant, on reprend la valeur optimisée de la ligne du dessus
                current_value = previous_optimized_value
            # on enregistre le resultat dans la matrice
            matrix[i][j] = current_value
    return matrix


def retrieve_selected_actions(matrix, amount, selected_actions):
    # la valeur optimal est matrix[-1][-1] ; on retrace les actions choisies en remontant la matrice
    # afin de retrouver les éléments en fonction de la somme
    remaining_amount = amount  # montant restant
    action_number = len(selected_actions)  # nombre d'actions à retrouver
    selected_actions_list = []  # stockage des actions sélectionnées

    # Boucle pour retrouver les éléments sélectionnés
    while remaining_amount >= 0 and action_number > 0:
        current_action = selected_actions[action_number - 1]  # Action actuelle
        action_cost = current_action[1]  # Coût de l'action actuelle
        action_benefit = current_action[2]  # Bénéfice de l'action actuelle

        # Vérifier si l'action actuelle a été incluse dans la solution optimale
        value_with_current_action = matrix[action_number - 1][remaining_amount - action_cost] + action_benefit
        value_without_current_action = matrix[action_number][remaining_amount]
        # si c'est le cas :
        if value_with_current_action == value_without_current_action:
            selected_actions_list.append(current_action)  # Ajouter l'action à la sélection
            remaining_amount -= action_cost  # réduit le montant restant du coût de l'action
        action_number -= 1  # on passe à l'action précédente

    return selected_actions_list


@measure_memory
@measure_time
def optimized_algo(amount, selected_actions):
    # initialisation de la matrice 💊😎
    matrix = initialize_matrix(amount, selected_actions)
    # remplissage de la matrice
    matrix = calculate_optimized_values(matrix, amount, selected_actions)
    # on retrouve les actions sélectionnées en remontant la matrice
    selected_actions_list = retrieve_selected_actions(matrix, amount, selected_actions)

    return matrix, matrix[-1][-1], selected_actions_list


def get_matrix_size(matrix):
    nombre_lignes = len(matrix)
    nombre_colonnes = len(matrix[0])
    return nombre_lignes * nombre_colonnes


def calculate_metrics(result_action_list, benefice_total):
    cout_total = sum([i[1] for i in result_action_list])
    plus_value_pourcent = (benefice_total / cout_total) * 100
    return cout_total, plus_value_pourcent


def remove_incorrect_values(data):
    # supprime les actions dans le cas ou où la valeur (prix, indice 1) est inférieure ou égale à 0
    cleaned_data = [action for action in data if action[1] > 0]
    print(f"nombre de valeurs inférieur ou égale à 0 supprimée : {len(data)-len(cleaned_data)}")
    return cleaned_data

def remove_duplicates(data):
    # on compte l'occurrence de chaque nom d'action
    action_names = [action[0] for action in data]
    name_counts = Counter(action_names)

    # on liste les actions qui sont présentes plus d'une fois
    duplicates = {name for name, count in name_counts.items() if count > 1}
    print(f"actions en doublon supprimées : {duplicates}")

    # on supprime TOUS ces doublons (la valeur en doublon et le doublon et pas uniquement le doublon)
    cleaned_data = [action for action in data if action[0] not in duplicates]
    return cleaned_data


def multiply_values(data, multiply_factor):
    # multiplie les coûts et les bénéfices par le facteur de multiplication
    data = [(action[0], action[1] * multiply_factor, action[2] * multiply_factor) for action in data]
    return data


def convert_to_int(data):
    # Convertit les valeurs mises à l'échelle en entiers
    data = [(action[0], int(action[1]), int(action[2])) for action in data]
    return data


def divide_values(data, divide_factor):
    # divise les valeurs après le traitement pour retrouver les valeurs originales
    data = [(action[0], action[1] / divide_factor, action[2] / divide_factor) for action in data]
    return data


def clean_dataset(data):
    data = remove_incorrect_values(data)
    data = remove_duplicates(data)
    # multiplication par 100
    data = multiply_values(data, MULTIPLIER_FACTOR)
    data = convert_to_int(data)
    return data


@redir_stdout_to_file(FILE_PATH_OUTPUT)
def main(MAX_AMOUNT, data):

    print(f"taille du jeu de données d'origine : {len(data)}")
    cleaned_data = clean_dataset(data)

    matrix, benefice_total, result_action_list = optimized_algo(MAX_AMOUNT*MULTIPLIER_FACTOR, cleaned_data)

    # on divise par 100 après traitement pour retrouver les valeurs initiales
    benefice_total = benefice_total / MULTIPLIER_FACTOR

    result_action_list = divide_values(result_action_list, MULTIPLIER_FACTOR)
    cout_total, plus_value_pourcent = calculate_metrics(result_action_list, benefice_total)

    print(f"Liste des actions choisies : {result_action_list}")
    print(TEXT_SEPARATOR)
    print(f"coût total de l'opération : {cout_total}€")
    print(f"gains : {benefice_total:.2f}€")
    print(f"plus value de : {plus_value_pourcent:.2f}%")
    print(TEXT_SEPARATOR)
    print(f"Nombre d'éléments calculés (taille matrice) : {get_matrix_size(matrix)}")


main(MAX_AMOUNT, data)
