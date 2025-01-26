import numpy as np
import pandas as pd
import time
import os
import re

def save_output_file_with_fixed_cost(costs, 
                                     depot_fixed_costs, 
                                     transport_fixed_costs, 
                                     allocation, 
                                     supply, 
                                     demand, 
                                     method_name, 
                                     instance_name):


    variable_cost = np.sum(costs * allocation)


    cost_fix_d2r = np.sum(transport_fixed_costs * (allocation > 0))

    used_depot = (np.sum(allocation, axis=1) > 0)
    cost_fix_opd = np.sum(depot_fixed_costs * used_depot)

    total_cost = int(variable_cost + cost_fix_opd + cost_fix_d2r)

    Xjk_rows = "\n".join(
        " ".join(f"{int(val)}" for val in row) for row in allocation
    )
    Uj = [int(u) for u in used_depot]
    Dk_str = " ".join(str(dk) for dk in demand)

  
    Fj_str = " ".join(str(fj) for fj in depot_fixed_costs)

    output = (
        f"Fj=\t\t [{Fj_str}]\n"
        f"Xjk=\n{Xjk_rows}\n"
        f"Uj=\t\t {Uj}\n"
        f"Dk=\t\t [{Dk_str}]\n"
        f"Optim        = {total_cost}\n"
        f"Cost fix OpD = {int(cost_fix_opd)}\n"
        f"Cost     D2R = {int(variable_cost)}\n"
        f"Cost fix D2R = {int(cost_fix_d2r)}\n"
    )

    output_dir = "Lab_FCD_FCR_solved"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"{instance_name}_{method_name}.txt"
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write(output)

    return total_cost, cost_fix_opd, variable_cost, cost_fix_d2r


def solve_instance_with_fixed_cost(costs, 
                                   capacity, 
                                   demand, 
                                   depot_fixed_costs, 
                                   transport_fixed_costs, 
                                   method_name, 
                                   instance_name, 
                                   approach):

    d = len(capacity)
    r = len(demand)    
    allocation = np.zeros((d, r), dtype=int)
    capacity_left = capacity.copy().astype(int)
    demand_left = demand.copy().astype(int)
    opened = [False] * d
    iterations = 0

    start_time = time.perf_counter()

    def approximate_unit_cost(j, k):
       
        base_cost = costs[j, k]
        arc_fixed = transport_fixed_costs[j, k]
        if not opened[j]:
            dep_fixed = depot_fixed_costs[j]
        else:
            dep_fixed = 0

        if approach == "simple_sum":
            return base_cost + arc_fixed + dep_fixed

        elif approach == "total_cost":
            possible_qty = min(capacity_left[j], demand_left[k])
            return base_cost * possible_qty + arc_fixed + dep_fixed

        elif approach == "distributed_cost":
            denom = max(demand_left[k], 1)
            return base_cost + (arc_fixed / denom) + (dep_fixed / denom)

        else:
            raise ValueError(f"Unknown approach: {approach}")

    if method_name == "rm":  
        for j in range(d):
            while capacity_left[j] > 0 and np.sum(demand_left) > 0:
                min_cost = float('inf')
                min_k = -1
                for k in range(r):
                    if demand_left[k] > 0:
                        cost_jk = approximate_unit_cost(j, k)
                        if cost_jk < min_cost:
                            min_cost = cost_jk
                            min_k = k
                if min_k == -1:
                    break
                alloc = min(capacity_left[j], demand_left[min_k])
                allocation[j, min_k] = alloc
                capacity_left[j] -= alloc
                demand_left[min_k] -= alloc
                if not opened[j] and alloc > 0:
                    opened[j] = True
                iterations += 1

    elif method_name == "mm":
        while np.sum(capacity_left) > 0 and np.sum(demand_left) > 0:
            min_cost = float('inf')
            min_j, min_k = -1, -1
            for j in range(d):
                if capacity_left[j] <= 0:
                    continue
                for k in range(r):
                    if demand_left[k] > 0:
                        cost_jk = approximate_unit_cost(j, k)
                        if cost_jk < min_cost:
                            min_cost = cost_jk
                            min_j, min_k = j, k
            if min_j == -1 or min_k == -1:
                break
            alloc = min(capacity_left[min_j], demand_left[min_k])
            allocation[min_j, min_k] = alloc
            capacity_left[min_j] -= alloc
            demand_left[min_k] -= alloc
            if not opened[min_j] and alloc > 0:
                opened[min_j] = True
            iterations += 1

    elif method_name == "vam":
        while np.sum(capacity_left) > 0 and np.sum(demand_left) > 0:
            penalties = []

            for j in range(d):
                if capacity_left[j] > 0:
                    row_costs = [(approximate_unit_cost(j, k), k) 
                                 for k in range(r) if demand_left[k] > 0]
                    if len(row_costs) > 1:
                        sorted_row_costs = sorted(row_costs, key=lambda x: x[0])
                        # penalty = 2nd cheapest - cheapest
                        penalty = sorted_row_costs[1][0] - sorted_row_costs[0][0]
                        penalties.append((penalty, j, 'row'))

            for k in range(r):
                if demand_left[k] > 0:
                    col_costs = [(approximate_unit_cost(j, k), j) 
                                 for j in range(d) if capacity_left[j] > 0]
                    if len(col_costs) > 1:
                        sorted_col_costs = sorted(col_costs, key=lambda x: x[0])
                        penalty = sorted_col_costs[1][0] - sorted_col_costs[0][0]
                        penalties.append((penalty, k, 'col'))

            if not penalties:
                break

            penalties.sort(reverse=True, key=lambda x: x[0])
            max_penalty, idx, axis = penalties[0]

            if axis == 'row':
                possible_cells = [(approximate_unit_cost(idx, k), k) 
                                  for k in range(r) if demand_left[k] > 0]
                if not possible_cells:
                    continue
                _, min_k = min(possible_cells, key=lambda x: x[0])
                alloc = min(capacity_left[idx], demand_left[min_k])
                allocation[idx, min_k] += alloc
                capacity_left[idx] -= alloc
                demand_left[min_k] -= alloc
                if not opened[idx] and alloc > 0:
                    opened[idx] = True

            else:  
                possible_cells = [(approximate_unit_cost(j, idx), j) 
                                  for j in range(d) if capacity_left[j] > 0]
                if not possible_cells:
                    continue
                _, min_j = min(possible_cells, key=lambda x: x[0])
                alloc = min(capacity_left[min_j], demand_left[idx])
                allocation[min_j, idx] += alloc
                capacity_left[min_j] -= alloc
                demand_left[idx] -= alloc
                if not opened[min_j] and alloc > 0:
                    opened[min_j] = True

            iterations += 1

        for j in range(d):
            for k in range(r):
                if capacity_left[j] > 0 and demand_left[k] > 0:
                    alloc = min(capacity_left[j], demand_left[k])
                    allocation[j, k] += alloc
                    capacity_left[j] -= alloc
                    demand_left[k] -= alloc

    else:
        raise ValueError(f"Unknown method_name: {method_name}")

    runtime = time.perf_counter() - start_time

    (total_cost, cost_fix_opd, var_cost, cost_fix_d2r) = save_output_file_with_fixed_cost(
        costs, 
        depot_fixed_costs, 
        transport_fixed_costs, 
        allocation, 
        capacity, 
        demand, 
        method_name, 
        instance_name
    )

    demand_satisfied = True
    for k in range(len(demand)):
        if np.sum(allocation[:, k]) != demand[k]:
            demand_satisfied = False
            break
    solved_status = "Solved" if demand_satisfied else "Not Solved"

    return instance_name, total_cost, iterations, runtime, solved_status


def process_all_instances_with_fixed_cost():
   
    methods = ["rm", "mm", "vam"]
    approaches = ["simple_sum", "total_cost", "distributed_cost"]
    results = {method: {approach: [] for approach in approaches} for method in methods}

    input_dir = "Lab_FCD_FCR_instances"
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".dat"):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r") as f:
                data = f.read()

            instance_name = re.search(r'instance_name\s*=\s*"([^"]+)";', data).group(1)
            d = int(re.search(r'd\s*=\s*(\d+);', data).group(1))
            r = int(re.search(r'r\s*=\s*(\d+);', data).group(1))

            SCj_str = re.search(r'SCj\s*=\s*\[([^\]]+)\];', data).group(1)
            SCj = list(map(int, SCj_str.split()))

            Fj_str = re.search(r'Fj\s*=\s*\[([^\]]+)\];', data).group(1)
            Fj = list(map(int, Fj_str.replace(",", " ").split()))

            Dk_str = re.search(r'Dk\s*=\s*\[([^\]]+)\];', data).group(1)
            Dk = list(map(int, Dk_str.split()))

            Cjk_data_section = re.search(r'Cjk\s*=\s*\[\[([\s\S]+?)\]\];', data).group(1)
            Cjk_cleaned = Cjk_data_section.replace("]", "").replace("[", "").replace("\n", " ").strip()
            Cjk_numbers = list(map(int, Cjk_cleaned.split()))
            Cjk = np.array(Cjk_numbers).reshape((d, r))

            Fjk_data_section = re.search(r'Fjk\s*=\s*\[\[([\s\S]+?)\]\];', data).group(1)
            Fjk_cleaned = Fjk_data_section.replace("]", "").replace("[", "").replace("\n", " ").strip()
            Fjk_numbers = list(map(int, Fjk_cleaned.split()))
            Fjk = np.array(Fjk_numbers).reshape((d, r))

            for method in methods:
                for approach in approaches:
                    try:
                        instance_result = solve_instance_with_fixed_cost(
                            costs=Cjk,
                            capacity=np.array(SCj),
                            demand=np.array(Dk),
                            depot_fixed_costs=np.array(Fj),
                            transport_fixed_costs=Fjk,
                            method_name=method,
                            instance_name=instance_name,
                            approach=approach
                        )
                        results[method][approach].append(instance_result)
                    except Exception as e:
                        print(f"Error on {instance_name}, method={method}, approach={approach}: {e}")

    for method, approach_dict in results.items():
        for approach, result_data in approach_dict.items():
            if result_data:
                df = pd.DataFrame(result_data, 
                                  columns=["Instance", 
                                           "Optimal (Heuristic)", 
                                           "Iterations", 
                                           "Runtime (sec)", 
                                           "Solved Status"])
                output_file = f"Lab_FCD_FCR_solved_{method.upper()}_{approach.upper()}.xlsx"
                df.to_excel(output_file, index=False)
                print(f"Results for {method.upper()} + {approach.upper()} saved to {output_file}")


if __name__ == "__main__":
    process_all_instances_with_fixed_cost()
