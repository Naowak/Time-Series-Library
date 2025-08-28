import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import ast
import argparse
from typing import Dict, List, Tuple, Any

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

def extract_anomaly_detection_results(base_dir: Path) -> List[Dict]:
    """Extrait tous les résultats de détection d'anomalies."""
    filename = base_dir / "result_anomaly_detection.txt"
    results = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Pattern pour extraire les informations de chaque tâche
        pattern = r'anomaly_detection_(\w+)_EST.*?Accuracy : ([\d.]+), Precision : ([\d.]+), Recall : ([\d.]+), F-score : ([\d.]+)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            dataset, accuracy, precision, recall, f_score = match
            results.append({
                'task_type': 'anomaly_detection',
                'dataset': dataset,
                'task_name': f'anomaly_detection_{dataset}',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f_score': float(f_score),
                'main_metric': float(f_score)
            })
    except FileNotFoundError:
        print(f"⚠️  Fichier {filename} non trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des résultats de détection d'anomalies: {e}")
    
    return results

def extract_imputation_results(base_dir: Path) -> List[Dict]:
    """Extrait tous les résultats d'imputation."""
    filename = base_dir / "result_imputation.txt"
    results = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Pattern pour extraire les informations de chaque tâche
        pattern = r'imputation_(\w+)_mask_([\d.]+)_EST.*?mse:([\d.]+), mae:([\d.]+)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            dataset, mask_ratio, mse, mae = match
            results.append({
                'task_type': 'imputation',
                'dataset': dataset,
                'mask_ratio': float(mask_ratio),
                'task_name': f'imputation_{dataset}_mask_{mask_ratio}',
                'mse': float(mse),
                'mae': float(mae),
                'main_metric': float(mse)
            })
    except FileNotFoundError:
        print(f"⚠️  Fichier {filename} non trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des résultats d'imputation: {e}")
    
    return results

def extract_long_term_forecast_results(base_dir: Path) -> List[Dict]:
    """Extrait tous les résultats de prévision à long terme."""
    filename = base_dir / "result_long_term_forecast.txt"
    results = []
    seen_tasks = set()  # Pour éviter les doublons
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Pattern plus flexible pour extraire les informations
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('long_term_forecast_'):
                # Extraire le nom de la tâche complet
                full_task_name = line
                
                # Extraire dataset et horizon de prédiction pour créer un nom simplifié
                pattern = r'long_term_forecast_(\w+)_\d+_(\d+)_EST'
                match = re.search(pattern, full_task_name)
                
                if match and i + 1 < len(lines):
                    dataset = match.group(1)
                    horizon = int(match.group(2))
                    
                    # Créer un nom de tâche simplifié (sans les paramètres du modèle)
                    simple_task_name = f'long_term_forecast_{dataset}_{horizon}'
                    
                    # Vérifier si cette tâche a déjà été traitée
                    if simple_task_name not in seen_tasks:
                        seen_tasks.add(simple_task_name)
                        
                        # Ligne suivante contient les métriques
                        metrics_line = lines[i + 1].strip()
                        mse_match = re.search(r'mse:([\d.]+)', metrics_line)
                        mae_match = re.search(r'mae:([\d.]+)', metrics_line)
                        
                        if mse_match and mae_match:
                            results.append({
                                'task_type': 'long_term_forecast',
                                'dataset': dataset,
                                'horizon': horizon,
                                'task_name': simple_task_name,
                                'full_task_name': full_task_name,  # Garder le nom complet pour référence
                                'mse': float(mse_match.group(1)),
                                'mae': float(mae_match.group(1)),
                                'main_metric': float(mse_match.group(1))
                            })
                    i += 2
                else:
                    i += 1
            else:
                i += 1
                
    except FileNotFoundError:
        print(f"⚠️  Fichier {filename} non trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des résultats de prévision à long terme: {e}")
    
    return results

def extract_classification_results(log_dir: Path) -> List[Dict]:
    """Extrait tous les résultats de classification."""
    results = []
    
    try:
        log_file = find_log_file(log_dir, 'classif_')
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Rechercher toutes les sections de test avec un pattern plus robuste
        # Pattern pour capturer le nom de dataset dans la section de test
        test_sections = re.findall(r'>>>>>>>testing : classification_([^_<]+)_EST.*?<<<.*?accuracy:([\d.]+)', content, re.DOTALL)
        
        for dataset, accuracy in test_sections:
            # Nettoyer le nom du dataset (enlever les tirets et caractères spéciaux si nécessaire)
            clean_dataset = dataset.replace('-', '_')
            
            results.append({
                'task_type': 'classification',
                'dataset': clean_dataset,
                'task_name': f'classification_{clean_dataset}',
                'accuracy': float(accuracy),
                'main_metric': float(accuracy)
            })
            
    except FileNotFoundError:
        print(f"⚠️  Fichier de log de classification non trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des résultats de classification: {e}")
    
    return results

def extract_short_term_forecast_results(log_dir: Path) -> List[Dict]:
    """Extrait tous les résultats de prévision à court terme."""
    results = []
    
    try:
        log_file = find_log_file(log_dir, 'short_term_forecast_')
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Rechercher toutes les sections de test
        sections = re.split(r'>>>>>>>testing : (short_term_forecast_m4_\w+_EST.*?)<<<<<<<<<', content)
        
        for i in range(1, len(sections), 2):
            task_name = sections[i]
            section_content = sections[i + 1] if i + 1 < len(sections) else ""
            
            # Extraire le type de série temporelle (Yearly, Quarterly, Monthly, Others)
            dataset_match = re.search(r'short_term_forecast_m4_(\w+)_EST', task_name)
            if not dataset_match:
                continue
                
            dataset = dataset_match.group(1)
            
            # Chercher les métriques OWA, SMAPE, MAPE, MASE
            metrics = {}
            for metric_name in ['owa', 'smape', 'mape', 'mase']:
                pattern = f"{metric_name}: ({{.*?}})"
                match = re.search(pattern, section_content)
                if match:
                    try:
                        dict_str = match.group(1).replace("'", '"')
                        metric_dict = ast.literal_eval(dict_str)
                        metrics[metric_name] = metric_dict
                    except:
                        continue
            
            # Créer une entrée pour cette tâche
            if 'owa' in metrics:
                task_result = {
                    'task_type': 'short_term_forecast',
                    'dataset': dataset,
                    'task_name': f'short_term_forecast_m4_{dataset}',
                    'main_metric': metrics['owa'].get('Average', 0)
                }
                
                # Ajouter toutes les métriques détaillées
                for metric_name, metric_dict in metrics.items():
                    for category, value in metric_dict.items():
                        task_result[f'{metric_name}_{category.lower()}'] = value
                
                results.append(task_result)
                
    except FileNotFoundError:
        print(f"⚠️  Fichier de log de prévision à court terme non trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des résultats de prévision à court terme: {e}")
    
    return results

def create_detailed_results_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """Crée un DataFrame détaillé avec tous les résultats."""
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Réorganiser les colonnes pour plus de clarté
    base_columns = ['task_type', 'task_name', 'dataset', 'main_metric']
    other_columns = [col for col in df.columns if col not in base_columns]
    
    return df[base_columns + sorted(other_columns)]

def save_results_to_csv(df: pd.DataFrame, run_name: str, output_dir: Path = None):
    """Sauvegarde les résultats dans un fichier CSV."""
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_file = output_dir / f"detailed_results_{run_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Résultats sauvegardés dans: {output_file}")
    return output_file

def print_summary_statistics(df: pd.DataFrame):
    """Affiche des statistiques résumées par type de tâche."""
    print("\n" + "="*80)
    print("📊 STATISTIQUES RÉSUMÉES PAR TYPE DE TÂCHE")
    print("="*80)
    
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        print(f"\n🔹 {task_type.upper().replace('_', ' ')}")
        print(f"   Nombre de tâches: {len(task_df)}")
        print(f"   Métrique principale moyenne: {task_df['main_metric'].mean():.4f}")
        print(f"   Métrique principale médiane: {task_df['main_metric'].median():.4f}")
        print(f"   Écart-type: {task_df['main_metric'].std():.4f}")
        
        # Afficher quelques exemples de noms de tâches pour vérifier
        print(f"   Exemples de tâches:")
        for i, task_name in enumerate(task_df['task_name'].head(3)):
            print(f"     - {task_name}")
        if len(task_df) > 3:
            print(f"     ... et {len(task_df) - 3} autres")

def main():
    """
    Fonction principale qui parse les arguments et lance l'extraction détaillée.
    """
    parser = argparse.ArgumentParser(
        description="Extrait les résultats détaillés pour toutes les tâches individuelles."
    )
    parser.add_argument(
        "run_directory", 
        type=str, 
        help="Le nom du dossier contenant les sous-dossiers 'logs' et 'results' (ex: run_1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dossier de sortie pour le fichier CSV (défaut: répertoire courant)"
    )
    args = parser.parse_args()

    # Construction des chemins
    base_path = Path(args.run_directory)
    log_dir = base_path / "logs"
    results_dir = base_path
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Validation des chemins
    if not base_path.is_dir():
        print(f"❌ Erreur: Le dossier spécifié '{base_path}' n'a pas été trouvé.")
        return

    print(f"🔍 Extraction des résultats détaillés pour: {base_path.name}")
    print("-" * 60)

    # Extraction des résultats pour chaque type de tâche
    all_results = []
    
    print("📋 Extraction des résultats de détection d'anomalies...")
    all_results.extend(extract_anomaly_detection_results(results_dir))
    
    print("📋 Extraction des résultats d'imputation...")
    all_results.extend(extract_imputation_results(results_dir))
    
    print("📋 Extraction des résultats de prévision à long terme...")
    all_results.extend(extract_long_term_forecast_results(results_dir))
    
    print("📋 Extraction des résultats de classification...")
    all_results.extend(extract_classification_results(log_dir))
    
    print("📋 Extraction des résultats de prévision à court terme...")
    all_results.extend(extract_short_term_forecast_results(log_dir))

    if not all_results:
        print("❌ Aucun résultat trouvé!")
        return

    # Création du DataFrame
    print(f"\n📊 Création du DataFrame avec {len(all_results)} tâches...")
    df = create_detailed_results_dataframe(all_results)
    
    # Sauvegarde en CSV
    csv_file = save_results_to_csv(df, base_path.name, output_dir)
    
    # Affichage des statistiques
    print_summary_statistics(df)
    
    # Affichage d'un aperçu du DataFrame
    print("\n" + "="*80)
    print("📋 APERÇU DES RÉSULTATS (5 premières lignes)")
    print("="*80)
    print(df.head().to_string(index=False))
    
    print(f"\n✅ Extraction terminée! {len(all_results)} tâches traitées.")
    print(f"📁 Fichier CSV généré: {csv_file}")

if __name__ == "__main__":
    main()