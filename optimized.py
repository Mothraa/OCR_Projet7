import json
import time
import sys
import tracemalloc
import gc
from functools import wraps
from collections import Counter
import configparser


""" Solution du problème du sac à dos (KP ou Knapsack)
    Algorithme par programmation dynamique

    Traitement préalable :
    - conversion des données d'origine en json (.data//actions_source.json)
    - conversion de la 3eme colonne (bénéfice à 2 ans en %)
    par la valeur brute du bénéfice en €. ATTENTION : arrondi à 2 digits
    => .data//fichier actions_tuples_benefice.json

    sources :
    https://en.wikipedia.org/wiki/Knapsack_problem
    https://fr.wikipedia.org/wiki/Probl%C3%A8me_du_sac_%C3%A0_dos
    analyse de la conso mémoire : https://www.ukonline.be/cours/python/opti/chapitre3-5
"""


class Config:
    """import configuration from ini file"""
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.MAX_AMOUNT = None
        self.MULTIPLIER_FACTOR = None
        self.TEXT_SEPARATOR = None
        self.NB_MEM_STATS = None
        self.FILE_PATH_INPUT = None
        self.FILE_PATH_OUTPUT = None
        self.load()

    def load(self):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        self.MAX_AMOUNT = config.getint('DEFAULT', 'MAX_AMOUNT')
        self.MULTIPLIER_FACTOR = config.getint('DEFAULT', 'MULTIPLIER_FACTOR')
        self.TEXT_SEPARATOR = config.get('DEFAULT', 'TEXT_SEPARATOR')
        self.NB_MEM_STATS = config.getint('DEFAULT', 'NB_MEM_STATS')
        self.FILE_PATH_INPUT = config.get('DEFAULT', 'FILE_PATH_INPUT')
        self.FILE_PATH_OUTPUT = config.get('DEFAULT', 'FILE_PATH_OUTPUT')


config = Config()


class MeasurePerformance():
    @staticmethod
    def measure_memory(func):
        """measure memory usage"""
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
            print(config.TEXT_SEPARATOR)
            # on affiche les 3 consommations de mémoire les plus significatives
            for stat in top_stats[:config.NB_MEM_STATS]:
                print(stat)
            print(config.TEXT_SEPARATOR)
            return result
        return wrapper

    @staticmethod
    def measure_time(func):
        """measure execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(config.TEXT_SEPARATOR)
            print(f"temps d'éxecution : {execution_time:.4f} secondes")
            return result
        return wrapper

    @staticmethod
    def redir_stdout_to_file(file_path):
        """redirection of stdout terminal to file"""
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


class DataTransformer():
    """class to handle loading and pre-processing data"""
    def __init__(self, filepath, multiply_factor) -> None:
        self.filepath = filepath
        self.multiply_factor = multiply_factor
        self.data = self.load_data()
        print(f"taille du jeu de données d'origine : {len(self.data)}")

    def load_data(self):
        """load data from json file"""
        with open(self.filepath, 'r') as file:
            return json.load(file)

    def remove_incorrect_values(self, data):
        """remove actions where price is =< 0"""
        cleaned_data = [action for action in data if action[1] > 0]
        print(f"nombre de valeurs inférieur ou égale à 0 supprimée : {len(data)-len(cleaned_data)}")
        return cleaned_data

    def remove_duplicates(self, data):
        """remove duplicates actions"""
        # on compte l'occurrence de chaque nom d'action
        action_names = [action[0] for action in data]
        name_counts = Counter(action_names)

        # on liste les actions qui sont présentes plus d'une fois
        duplicates = {name for name, count in name_counts.items() if count > 1}
        print(f"actions en doublon supprimées : {duplicates}")

        # on supprime TOUS ces doublons (la valeur en doublon et le doublon et pas uniquement le doublon)
        data = [action for action in data if action[0] not in duplicates]
        return data

    def multiply_values(self, data):
        """multiply cost and benefices by one factor"""
        data = [(action[0], action[1] * self.multiply_factor, action[2] * self.multiply_factor) for action in data]
        return data

    def convert_to_int(self, data):
        """convert cost and benefices from float to int"""
        data = [(action[0], int(action[1]), int(action[2])) for action in data]
        return data

    def divide_values(self, data):
        """divide cost and benefices by one factor to retrieve original values"""
        data = [(action[0], action[1] / self.multiply_factor, action[2] / self.multiply_factor) for action in data]
        return data

    def clean_dataset(self):
        data = self.remove_incorrect_values(self.data)
        data = self.remove_duplicates(data)
        data = self.multiply_values(data)
        data = self.convert_to_int(data)
        return data


class KnapsackOptimizedSolver():
    """For solving the knapsack problem with optimized dynamic solving (exact solution)
    informations : https://en.wikipedia.org/wiki/Knapsack_problem
    """
    def __init__(self, max_amount, data):
        self.max_amount = max_amount
        self.data = data
        self.matrix = self.initialize_matrix()

    # @measure_memory
    def initialize_matrix(self):
        """initialisation de la matrice 💊😎"""
        # on initialise la matrice (2 listes imbriquées) à 0 (montant+1 * nombre d'actions+1)
        return [[0 for _ in range(self.max_amount + 1)] for _ in range(len(self.data) + 1)]  # self.selected_actions

    # @measure_memory
    # @measure_time
    def calculate_optimized_values(self):
        # on parcours chaque action
        for i in range(1, len(self.data) + 1):
            # pour chaque element on parcours la "capacité du sac" (montant) > les colonnes de la matrice
            for j in range(1, self.max_amount + 1):
                # Valeur optimisée sans inclure l'action actuelle
                previous_optimized_value = self.matrix[i-1][j]
                # si cout de l'element inférieur au montant(colonne), on peut mettre l'element dedans
                if self.data[i-1][1] <= j:
                    # bénéfice de l'action actuelle
                    current_benefit = self.data[i-1][2]
                    # valeur optimisée pour le montant restant
                    remaining_optimized_value = self.matrix[i-1][j - self.data[i-1][1]]
                    # valeur optimisée si on inclus l'action actuelle
                    new_optimized_value = current_benefit + remaining_optimized_value
                    # on prend la valeur max entre la valeur qui inclus l'action en cours et celle qui ne l'inclus pas
                    current_value = max(new_optimized_value, previous_optimized_value)
                else:  # si cout supérieur au montant, on reprend la valeur optimisée de la ligne du dessus
                    current_value = previous_optimized_value
                # on enregistre le resultat dans la matrice
                self.matrix[i][j] = current_value
        return self.matrix

    def retrieve_selected_actions(self):
        # la valeur optimal est matrix[-1][-1] ; on retrace les actions choisies en remontant la matrice
        # afin de retrouver les éléments en fonction de la somme
        remaining_amount = self.max_amount  # montant restant
        action_number = len(self.data)  # nombre d'actions à retrouver
        selected_actions_list = []  # stockage des actions sélectionnées

        # Boucle pour retrouver les éléments sélectionnés
        while remaining_amount >= 0 and action_number > 0:
            current_action = self.data[action_number - 1]  # Action actuelle
            action_cost = current_action[1]  # Coût de l'action actuelle
            action_benefit = current_action[2]  # Bénéfice de l'action actuelle

            # Vérifier si l'action actuelle a été incluse dans la solution optimale
            value_with_current_action = self.matrix[action_number - 1][remaining_amount - action_cost] + action_benefit
            value_without_current_action = self.matrix[action_number][remaining_amount]
            # si c'est le cas :
            if value_with_current_action == value_without_current_action:
                selected_actions_list.append(current_action)  # Ajouter l'action à la sélection
                remaining_amount -= action_cost  # réduit le montant restant du coût de l'action
            action_number -= 1  # on passe à l'action précédente

        return selected_actions_list

    @MeasurePerformance.measure_memory
    @MeasurePerformance.measure_time
    def run(self):
        # remplissage de la matrice
        self.matrix = self.calculate_optimized_values()
        # on retrouve les actions sélectionnées en remontant la matrice
        selected_actions_list = self.retrieve_selected_actions()
        return self.matrix, self.matrix[-1][-1], selected_actions_list


class Main():
    def __init__(self, max_amount, file_input, file_output):
        self.max_amount = max_amount
        self.file_input = file_input
        self.file_output = file_output

    def get_matrix_size(self, matrix):
        """return the matrix size (n x m)"""
        nombre_lignes = len(matrix)
        nombre_colonnes = len(matrix[0])
        return nombre_lignes * nombre_colonnes

    def calculate_metrics(self, result_action_list, benefice_total):
        """calculate of metrics"""
        cout_total = sum([i[1] for i in result_action_list])
        plus_value_pourcent = benefice_total / cout_total
        return cout_total, plus_value_pourcent

    @MeasurePerformance.redir_stdout_to_file(config.FILE_PATH_OUTPUT)
    def run(self):
        # instantiation et chargement des données
        self.data_transformer = DataTransformer(config.FILE_PATH_INPUT, config.MULTIPLIER_FACTOR)
        # nettoyage des données
        cleaned_data = self.data_transformer.clean_dataset()
        # instanciation et init de la matrice
        knapsack_solver = KnapsackOptimizedSolver(self.max_amount * config.MULTIPLIER_FACTOR, cleaned_data)
        # calcul des valeurs
        matrix, benefice_total, result_action_list = knapsack_solver.run()
        # on divise par 100 après traitement pour retrouver les valeurs initiales
        benefice_total /= config.MULTIPLIER_FACTOR

        result_action_list = self.data_transformer.divide_values(result_action_list)
        cout_total, plus_value_pourcent = self.calculate_metrics(result_action_list, benefice_total)

        print(f"Liste des actions choisies : {result_action_list}")
        print(config.TEXT_SEPARATOR)
        print(f"coût total de l'opération : {cout_total}€")
        print(f"gains : {benefice_total:.2f}€")
        print(f"plus value de : {100 * plus_value_pourcent:.2f}%")
        print(config.TEXT_SEPARATOR)
        print(f"Nombre d'éléments calculés (taille matrice) : {self.get_matrix_size(matrix)}")


app = Main(config.MAX_AMOUNT, config.FILE_PATH_INPUT, config.FILE_PATH_OUTPUT)
app.run()
