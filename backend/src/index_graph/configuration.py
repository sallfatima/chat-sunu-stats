# src/index_graph/configuration.py
from __future__ import annotations
from dataclasses import dataclass, field
from shared.configuration import BaseConfiguration
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_DOCS_FILE = "src/sample_docs.json"

@dataclass(kw_only=True)
class IndexConfiguration(BaseConfiguration):
    """Configuration pour l'indexation textuelle et visuelle améliorée."""

    # — Textuel (existant) —
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        metadata={"description": "Clé OpenAI pour embeddings textuels et extraction visuelle."}
    )
    pinecone_index: str = field(
        default="index-ansd",
        metadata={"description": "Nom de l'index Pinecone pour le texte et contenu visuel."}
    )

    # — Indexation visuelle (nouveau et amélioré) —
    chart_index_path: str = field(
        default_factory=lambda: os.getenv("CHART_INDEX_PATH", "charts_index.csv"),
        metadata={"description": "Chemin vers le CSV d'index des graphiques."}
    )
    
    table_index_path: str = field(
        default_factory=lambda: os.getenv("TABLE_INDEX_PATH", "tables_index.csv"),
        metadata={"description": "Chemin vers le CSV d'index des tableaux."}
    )
    
    images_dir: str = field(
        default_factory=lambda: os.getenv("IMAGES_DIR", "images"),
        metadata={"description": "Dossier racine des images extraites."}
    )
    
    tables_dir: str = field(
        default_factory=lambda: os.getenv("TABLES_DIR", "tables"),
        metadata={"description": "Dossier racine des tableaux extraits."}
    )

    # — Paramètres d'extraction visuelle —
    vision_model: str = field(
        default="gpt-4o-mini",
        metadata={"description": "Modèle OpenAI avec capacité vision pour extraction de texte des images."}
    )
    
    max_vision_retries: int = field(
        default=3,
        metadata={"description": "Nombre maximum de tentatives pour l'extraction de texte d'images."}
    )
    
    max_table_rows: int = field(
        default=20,
        metadata={"description": "Nombre maximum de lignes de tableau à indexer."}
    )

    # — Configuration d'indexation par batch —
    visual_batch_size: int = field(
        default=5,
        metadata={"description": "Taille des batches pour l'indexation des éléments visuels."}
    )
    
    enable_visual_indexing: bool = field(
        default=True,
        metadata={"description": "Activer l'indexation du contenu visuel (graphiques et tableaux)."}
    )
    
    # — Paramètres hérités et modifiés —
    text_embedding_model: str = field(
        default="text-embedding-3-small",
        metadata={"description": "Modèle OpenAI pour embeddings texte (utilisé pour tout le contenu)."}
    )
    
    # — Configuration Pinecone —
    pinecone_api_key: str = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", ""),
        metadata={"description": "Clé API Pinecone."}
    )
    
    pinecone_env: str = field(
        default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        metadata={"description": "Environnement Pinecone."}
    )

    def validate_visual_config(self) -> bool:
        """
        Valide la configuration pour l'indexation visuelle.
        
        Returns:
            True si la configuration est valide
        """
        if not self.enable_visual_indexing:
            return True
        
        issues = []
        
        # Vérifier les clés API
        if not self.api_key:
            issues.append("OPENAI_API_KEY manquante")
        
        if not self.pinecone_api_key:
            issues.append("PINECONE_API_KEY manquante")
        
        # Vérifier les fichiers d'index
        from pathlib import Path
        
        chart_path = Path(self.chart_index_path)
        if not chart_path.exists():
            issues.append(f"Fichier d'index graphiques non trouvé: {self.chart_index_path}")
        
        table_path = Path(self.table_index_path)
        if not table_path.exists():
            issues.append(f"Fichier d'index tableaux non trouvé: {self.table_index_path}")
        
        if issues:
            print("⚠️ Problèmes de configuration visuelle détectés:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        return True

    def get_visual_stats_summary(self) -> str:
        """
        Retourne un résumé de la configuration visuelle.
        
        Returns:
            Résumé de configuration
        """
        if not self.enable_visual_indexing:
            return "Indexation visuelle désactivée"
        
        from pathlib import Path
        import pandas as pd
        
        summary_parts = []
        
        # Compter les graphiques
        try:
            chart_path = Path(self.chart_index_path)
            if chart_path.exists():
                charts_df = pd.read_csv(chart_path)
                summary_parts.append(f"📊 {len(charts_df)} graphiques à indexer")
            else:
                summary_parts.append("📊 Aucun index de graphiques")
        except Exception:
            summary_parts.append("📊 Erreur lecture index graphiques")
        
        # Compter les tableaux
        try:
            table_path = Path(self.table_index_path)
            if table_path.exists():
                tables_df = pd.read_csv(table_path)
                summary_parts.append(f"📋 {len(tables_df)} tableaux à indexer")
            else:
                summary_parts.append("📋 Aucun index de tableaux")
        except Exception:
            summary_parts.append("📋 Erreur lecture index tableaux")
        
        # Configuration
        summary_parts.append(f"🔧 Modèle vision: {self.vision_model}")
        summary_parts.append(f"🔧 Batch size: {self.visual_batch_size}")
        
        return " | ".join(summary_parts)