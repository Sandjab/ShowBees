CONSTITUTION DES CHUNKS

La labelisation bee/nobee d'un segment se base sur un agrégat de la durée des sons extérieurs ne prenant en compte que les perturbations dont la durée unitaire est supérieure à un seuil passé en paramètre.

Si cet agrégat est non nul, le segment est considéré "nobee".

Cependant, de la façon dont c'est calculé (sur la longueur de l'évenement nobee, sans considérer son recouvrement avec un segment, on rejette trop de segments. Par exemple,avec un seuil à 0.5s

Sur des segments courts, cela a peu d'importance, mais sur des segments longs, il vaudrait sans doute mieux décider en se basant sur un % de perturbation total (durée cumulée de tous les nobee d'un segment quelle que soit leur durée / durée du segment)

---------------------------------------------------------------------------------------------------------------------



Sur mon laptop (Core i7, SSD, 8Go RAM), avec le code hérité de Nolasco, le chargement du dataset de l'article met plus de 4h30 (dont 4h rien que pour le gros mp3 du dataset) pour traiter 48 fichiers et constituer les 24816 chunks de 1 seconde.
La librairie utilisée (Librosa) pour le chargement des fichiers sons étant peu performante, on pourrait envisager de basculer sur une librairie plus performante (pydub, pySox)... 
Mais l'erreur est en réalité conceptuelle: en effet, on invoque en boucle librosa.core.load sur un même fichier pour chaque chunk de 1s, avec un temps d'exécution unitaire en O(n) (puisqu'il faut décoder les k premières secondes du fichier pour découper le chunk k+1.
Résultat des courses, un temps d'éxecution en O(n^2) pour le slicing d'un seul fichier source...
C'était sans doute peu visible sur les chunks de plus longue durée (60s) utilisés par Nolasco, mais c'est rédhibitoire sur des chunks de 1s


---------------------------------------------------------------------------------------------------------------------
GENERAL
STRUCTURANT
 - Refactoring lourd du code hérité de Nolasco
 - Structuration 'opinionated' de l'arborescence
 - Passage en objet quand cela semblait pertinent
 - Gestion des datasets en base de données (SQLite3) pour rendre les manips plus aisées et le code plus concis
 
NON STRUCTURANT 
 - Passage systématique de os.path à pathlib
 - suppression de la gestion partielle des exceptions: l'environnement devient suffisament maitrisé pour qu'on puisse se permettre de juste interrompre le traitement
 - a contrario, ajout systématique d'asserts de sanity checks 

DETAIL
PREPROCESSING
 - Amélioration massive des performances (x15) grâce au chargement et resampling du fichier en une seule fois, puis constitution des chunks en mémoire.
 - Ajout d'un paramètre overlap, permettant de générer des chunks en chevauchement
 - Remplacement des threshold (durée minimale de bruit externe par un ratio de perturbation (% de nobee d'un segment) qui pourra ensuite être utilisé par seuillage
 - génération d'une base de données de référence par dataset, point d'entrée unique des autres traitement
 - Si un chunk n'atteint pas la durée spécifiée (cas potentiel du dernier chunk d'un fichier source), il n'est tout simplement pas généré. La complétion de ce type de chunk par mirroring a en effet peu d'intérêt pour des durées courtes (où l'on dispose de suffisament de fichiers) , et elle introduit un artefact non maitrisé dans le jeu de test. Si besoin, cela pourra toujour être rajouté ultérieurement.


     
 