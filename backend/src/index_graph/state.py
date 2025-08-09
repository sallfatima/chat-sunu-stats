"""
Gestion de l'état d'indexation (textuelle et visuelle) selon SOLID.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass(kw_only=True)
class InputState:
    """État des entrées pour l'indexation locale de documents."""

    pdf_root: str = field(
        default="",
        metadata={"description": "Dossier racine des PDFs à indexer."}
    )

@dataclass(kw_only=True)
class IndexState(InputState):
    """
    Structure de l'état pour l'indexation de documents locaux,
    incluant le suivi des fichiers traités et des images indexées.
    """
    status: str = field(
        default="",
        metadata={"description": "Statut actuel de l'opération d'indexation."}
    )
    processed_text_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs traités en texte."}
    )
    failed_text_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs dont le traitement textuel a échoué."}
    )
    processed_image_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs dont les images ont été indexées."}
    )
    failed_image_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs dont l'indexation visuelle a échoué."}
    )
    total_text_chunks: int = field(
        default=0,
        metadata={"description": "Nombre total de segments textuels indexés."}
    )
    total_text_files_found: int = field(
        default=0,
        metadata={"description": "Nombre total de fichiers PDF trouvés."}
    )
    total_images_indexed: int = field(
        default=0,
        metadata={"description": "Nombre total d'images indexées."}
    )
    last_indexed_text: datetime | None = field(
        default=None,
        metadata={"description": "Horodatage du dernier index textuel."}
    )
    last_indexed_image: datetime | None = field(
        default=None,
        metadata={"description": "Horodatage du dernier index visuel."}
    )
    
    # === NOUVEAUX CHAMPS POUR L'EXTRACTION VISUELLE ===
    visual_extraction_completed: bool = field(
        default=False,
        metadata={"description": "Indique si l'extraction du contenu visuel est terminée."}
    )
    
    total_images_extracted: int = field(
        default=0,
        metadata={"description": "Nombre total d'images extraites des PDFs."}
    )
    
    total_tables_extracted: int = field(
        default=0,
        metadata={"description": "Nombre total de tableaux extraits des PDFs."}
    )
    
    extraction_errors: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des erreurs rencontrées lors de l'extraction."}
    )
    
    # === MÉTHODES UTILITAIRES ===
    def mark_image_processed(self, image_path: str):
        """Marque une image comme traitée."""
        if image_path not in self.processed_image_files:
            self.processed_image_files.append(image_path)

    def mark_image_failed(self, image_path: str, error: str):
        """Marque une image comme échouée."""
        if image_path not in self.failed_image_files:
            self.failed_image_files.append(image_path)
            self.failed_image_files.append(f"Error: {error}")
            
    def add_extraction_error(self, error: str):
        """Ajoute une erreur d'extraction à la liste."""
        if error not in self.extraction_errors:
            self.extraction_errors.append(error)
            
    def mark_visual_extraction_completed(self, images_extracted: int = 0, tables_extracted: int = 0):
        """Marque l'extraction visuelle comme terminée avec les statistiques."""
        self.visual_extraction_completed = True
        self.total_images_extracted = images_extracted
        self.total_tables_extracted = tables_extracted
        self.last_indexed_image = datetime.utcnow()
        
    def get_visual_extraction_summary(self) -> dict:
        """Retourne un résumé de l'extraction visuelle."""
        return {
            "completed": self.visual_extraction_completed,
            "images_extracted": self.total_images_extracted,
            "tables_extracted": self.total_tables_extracted,
            "total_visual_elements": self.total_images_extracted + self.total_tables_extracted,
            "errors_count": len(self.extraction_errors),
            "last_extraction": self.last_indexed_image.isoformat() if self.last_indexed_image else None
        }