import json
import time

MAX_AMOUNT = 500  # montant max en euros par clients

# file_path = './/data//actions_tuples_decimal.json'
file_path = './/data//actions_tuples_benefice.json'
with open(file_path, 'r') as file:
    data = json.load(file)
print(data)
print("-"*20)


def brute_force(amount, data, selected_actions):
    if data:  # si data vide c'est qu'ils ont tous été passés en revue
        # data structure : ("nom action", cout_action, benefice_a_2_ans)
        # on prend le cas ou l'element courant n'est pas selectionné
        # on compare le cas ou on prend l'element (item_2, list_item_2) et le cas ou on ne le prend pas (item_1,...).
        # on appelle la fonction sans tenir compte de l'élément courant (indice 0)
        item_1, list_item_1 = brute_force(amount, data[1:], selected_actions)
        action = data[0]

        if action[1] <= amount:  # on regarde si on peut ajouter l'objet sans dépasser le montant max
            new_amount = amount - action[1]  # on déduit du total le cout de l'action que l'on vient d'ajouter
            item_2, list_item_2 = brute_force(new_amount, data[1:], selected_actions + [action])
            # data[1:] liste d'éléments privés du 1er car traités
            # quelle est la meilleure solution ? quand on selectionne l'element courant ou quand on l'ajoute pas ?
            if item_1 > item_2:
                return item_1, list_item_1
            else:
                return item_2, list_item_2
        else:
            return item_1, list_item_1  # si on ne peut pas ajouter l'élément, on retourne le cas sans lui

    else:
        # si data est vide, on calcul le benefice et on retourne la liste des actions
        benefice = sum([i[2] for i in selected_actions])
        return benefice, selected_actions


selected_actions = []
# Initialiser le timer
start_time = time.time()
benefice_total, result_list = brute_force(MAX_AMOUNT, data, selected_actions)
end_time = time.time()
execution_time = end_time - start_time
print(f"{execution_time:.4f} secondes")
cout_total = sum([i[1] for i in result_list])
plus_value_pourcent = (benefice_total / cout_total)*100

print(f"Liste des actions : {result_list}")
print(f"coût total de l'opération : {cout_total}€")
print(f"gains : {benefice_total}€")
print(f"plus value de : {plus_value_pourcent:.2f}%")


print(f"Nombre d'éléments calculés : {2**len(data)}")
