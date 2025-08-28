import os
import re
import numpy as np
from pathlib import Path
import ast
import argparse # Module pour gérer les arguments de la ligne de commande

def find_log_file(log_dir: Path, prefix: str) -> Path:
    """
    Trouve le premier fichier log '.out' dans un répertoire qui commence par un préfixe donné.
    """
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Le dossier de logs '{log_dir}' n'existe pas.")
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix) and filename.endswith('.out'):
            return log_dir / filename
    raise FileNotFoundError(f"Aucun fichier '.out' avec le préfixe '{prefix}' trouvé dans '{log_dir}'")

def calculate_anomaly_detection(base_dir: Path, expected_count: int) -> float:
    """Calcule le F-score moyen pour la détection d'anomalies."""
    filename = base_dir / "result_anomaly_detection.txt"
    with open(filename, 'r') as f:
        text = f.read()
    f_scores = re.findall(r'F-score : ([\d.]+)', text)
    if expected_count is not None and len(f_scores) != expected_count:
        raise ValueError(f"Nombre de F-scores trouvé ({len(f_scores)}) incorrect. Attendu : {expected_count}. Current F-score: {np.mean(np.array(f_scores, dtype=float) if f_scores else 'nan')}")
    if not f_scores:
        raise ValueError("Aucun F-score trouvé dans le fichier de détection d'anomalies.")
    return np.mean(np.array(f_scores, dtype=float))

def calculate_imputation(base_dir: Path, expected_count: int) -> float:
    """Calcule le MSE moyen pour l'imputation."""
    filename = base_dir / "result_imputation.txt"
    with open(filename, 'r') as f:
        text = f.read()
    mses = re.findall(r'mse:([\d.]+)', text)
    if expected_count is not None and len(mses) != expected_count:
        raise ValueError(f"Nombre de MSE trouvé ({len(mses)}) incorrect. Attendu : {expected_count}. Current mse: {np.mean(np.array(mses, dtype=float) if mses else 'nan')}")
    if not mses:
        raise ValueError("Aucune valeur MSE trouvée dans le fichier d'imputation.")
    return np.mean(np.array(mses, dtype=float))

def calculate_long_term_forecast(base_dir: Path, expected_count: int) -> float:
    """Calcule le MSE moyen pour la prévision à long terme."""
    filename = base_dir / "result_long_term_forecast.txt"
    with open(filename, 'r') as f:
        text = f.read()
    mses = re.findall(r'mse:([\d.]+)', text)
    if expected_count is not None and len(mses) != expected_count:
        raise ValueError(f"Nombre de MSE trouvé ({len(mses)}) incorrect. Attendu : {expected_count}. Current mse: {np.mean(np.array(mses, dtype=float) if mses else 'nan')}")
    if not mses:
        raise ValueError("Aucune valeur MSE trouvée dans le fichier de prévision à long terme.")
    return np.mean(np.array(mses, dtype=float))

def calculate_classification(log_dir: Path, expected_count: int) -> float:
    """Calcule la précision (accuracy) moyenne pour la classification depuis les logs."""
    log_file = find_log_file(log_dir, 'classif_')
    with open(log_file, 'r') as f:
        text = f.read()
    accuracies = re.findall(r'accuracy:([\d.]+)', text)
    if expected_count is not None and len(accuracies) != expected_count:
        raise ValueError(f"Nombre de 'accuracy' trouvé ({len(accuracies)}) incorrect. Attendu : {expected_count}., Current accuracy: {np.mean(np.array(accuracies, dtype=float) if accuracies else 'nan')}")
    if not accuracies:
        raise ValueError("Aucune valeur 'accuracy' trouvée dans le log de classification.")
    return np.mean(np.array(accuracies, dtype=float))

def calculate_short_term_forecast(log_dir: Path, expected_count: int) -> float:
    """Calcule la moyenne de tous les scores OWA 'Average' dans le log de prévision à court terme."""
    
    log_file = find_log_file(log_dir, 'short_term_forecast_')
    owa_averages = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'owa:' in line:
                dict_str_match = re.search(r"(\{.*\})", line)
                if dict_str_match:
                    owa_dict = ast.literal_eval(dict_str_match.group(1).replace("'", '"'))
                    if 'Average' in owa_dict:
                        owa_averages.append(owa_dict['Average'])

    if not owa_averages:
        raise ValueError("Aucune valeur 'Average' trouvée dans les dictionnaires 'owa' du log.")
    
    if expected_count is not None and len(owa_averages) != expected_count:
        raise ValueError(f"Nombre de valeurs OWA 'Average' trouvé ({len(owa_averages)}) incorrect. Attendu : {expected_count}. Current OWA Average: {np.mean(np.array(owa_averages, dtype=float) if owa_averages else 'nan')}")
    
    return np.mean(np.array(owa_averages, dtype=float))

def main():
    """
    Fonction principale qui parse les arguments et lance les calculs.
    """
    # --- Configuration de l'analyseur d'arguments ---
    parser = argparse.ArgumentParser(
        description="Calcule les scores pour un dossier d'exécution (run) spécifique."
    )
    parser.add_argument(
        "run_directory", 
        type=str, 
        help="Le nom du dossier contenant les sous-dossiers 'logs' et 'results' (ex: run_1)"
    )
    args = parser.parse_args()

    # --- Construction des chemins ---
    base_path = Path(args.run_directory)
    log_dir = base_path / "logs"
    results_dir = base_path 
    # --- Validation des chemins ---
    if not base_path.is_dir():
        print(f"❌ Erreur: Le dossier spécifié '{base_path}' n'a pas été trouvé.")
        return

    scores = {}
    # Dictionnaire des tâches avec la fonction, le dossier requis et le nombre de résultats attendus
    tasks = {
        "Anomaly Detection (Mean F-score)": (calculate_anomaly_detection, results_dir, 5),
        "Imputation (Mean MSE)": (calculate_imputation, results_dir, 12),
        "Long-Term Forecast (Mean MSE)": (calculate_long_term_forecast, results_dir, 36),
        "Classification (Mean Accuracy)": (calculate_classification, log_dir, 10),
        "Short-Term Forecast (OWA Average)": (calculate_short_term_forecast, log_dir, 6) # Pas de compte attendu
    }

    for name, (func, data_dir, expected_count) in tasks.items():
        try:
            if expected_count is not None:
                # Appelle la fonction avec le nombre de résultats attendu
                scores[name] = func(data_dir, expected_count)
            else:
                # Appelle la fonction sans ce paramètre (cas de short_term_forecast)
                scores[name] = func(data_dir)
        except (FileNotFoundError, ValueError, KeyError, IsADirectoryError) as e:
            scores[name] = f"ERREUR: {e}"

    print(f"\n--- ✅ Résumé final des scores pour {base_path.name} ---\n")
    for name, score in scores.items():
        if isinstance(score, float):
            print(f" - {name:<35}: {score:.4f}")
        else:
            print(f" - {name:<35}: {score}")
    print("\n-------------------------------------------------")


if __name__ == "__main__":
    main()