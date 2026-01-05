# Use a pipeline as a high-level helper
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# 1. Load a pretrained model
embed = SentenceTransformer("all-MiniLM-L6-v2")


ner = pipeline("token-classification", model="dslim/bert-base-NER")




def build_summary(axes, profile_type, level):

    
    final_text = f"{profile_type} oriented profile"

    
    strong_axes = []
    for axis, value in axes.items():
        if value >= 3:
            strong_axes.append(axis)

   
    if "ml" in strong_axes:
        final_text += " with a specialization in Machine Learning"
    elif "cloud" in strong_axes:
        final_text += " with a strong Cloud orientation"
    elif "backend" in strong_axes:
        final_text += " specialized in Backend Development"


    if level == "advanced":
        final_text += " and solid professional experience"
    elif level == "junior":
        final_text += " at beginner level"

    return final_text



def extract(text):
    """
    Extrait les compétences (skills) du texte en utilisant la reconnaissance d'entités nommées (NER).
    Traite les tokens BERT WordPiece, regroupe les entités adjacentes et nettoie les résultats.
    
    Args:
        text (str): Le texte à analyser (ex: CV)
    
    Returns:
        list: Liste des compétences extraites et normalisées
    """
    results = ner(text)
    entities = [ent for ent in results if ent["score"] >= 0.60]

    # Reconstruction WordPiece
    buffer_word = None
    words = []
    prev_start = None
    prev_end = None

    for ent in entities:
        word = ent["word"]
        start = ent["start"]
        end = ent["end"]

        if word.startswith("##"):
            word_piece = word.lstrip("#")
            if buffer_word is not None:
                buffer_word += word_piece
        else:
            if buffer_word is not None:
                words.append({"word": buffer_word, "start": prev_start, "end": prev_end})
            buffer_word = word
            prev_start = start
            prev_end = end

        prev_end = end

    if buffer_word is not None:
        words.append({"word": buffer_word, "start": prev_start, "end": prev_end})

    # Regroupement des mots adjacents
    skills = []
    current = None
    previous_end = None

    for ent in words:
        word = ent["word"]
        start = ent["start"]
        end = ent["end"]

        if current is None:
            current = word
            previous_end = end
            continue

        between = text[previous_end:start]

        if between.strip() in ["", "-", "/"]:
            current = current + " " + word
        else:
            skills.append(current)
            current = word

        previous_end = end

    if current is not None:
        skills.append(current)

    # Nettoyage, normalisation et corrections
    skills_cleaned = []
    skills_seen = set()

    corrections = {
        "tenflow": "tensorflow",
        "tenflowai": "tensorflow",
        "djangoapi": "fastapi",
        "fast api": "fastapi",
        "pythn": "python",
        "ptorch": "pytorch",
        "djangoi": "django",
        "opena": "openai",
        "scikitlearn": "scikit-learn",
        "scikit learn": "scikit-learn",
        "open ai": "openai",
        "aws": "aws",
        "azure": "azure",
        "gcp": "gcp",
        "keras": "keras",
        "nlp": "natural language processing",
        "machine learning": "machine learning",
        "deep learning": "deep learning",
        "tensorflow": "tensorflow",
        "pytorch": "pytorch",
        "django": "django",
        "fastapi": "fastapi",
        "openai": "openai"
    }

    # Appliquer corrections et filtrer doublons
    for skill in skills:
        skill_norm = skill.strip().lower()
        skill_norm = corrections.get(skill_norm, skill_norm)
        if len(skill_norm) < 3 or skill_norm in skills_seen:
            continue
        if skill_norm in ["fast", "api"]:
            continue
        skills_cleaned.append(skill_norm)
        skills_seen.add(skill_norm)

    # Ajouter les compétences connues présentes dans le texte mais manquées par le NER
    known_skills = list(corrections.values())
    for skill in known_skills:
        if skill not in skills_seen and skill.lower() in text.lower():
            skills_cleaned.append(skill)
            skills_seen.add(skill)

    return skills_cleaned


def reference_profiles():
    """
    Crée les profils de référence avec leurs compétences associées et génère les vecteurs d'embedding.
    Chaque profil contient un type de professionnel (Backend, Data Scientist, etc.) et la liste de ses compétences.
    Les vecteurs d'embedding sont calculés pour permettre la comparaison de similarité cosinus.
    
    Returns:
        list: Liste de dictionnaires contenant 'name' (type de profil), 'skills' (compétences) et 'vector' (embedding)
    """
    profiles = []

    profiles.append({
        "name": "Backend Developer",
        "skills" : ["python", "django", "fastapi", "flask","rest api",
                   "sql", "postgresql","authentication", "docker"]
    })
                    

    profiles.append({
        "name": "Fullstack Developer",
        "skills": ["python","django","javascript","html", "css","react", "api", "postgresql", "git" ]
    })

    profiles.append({
        "name": "Data Analyst",
        "skills": ["python", "pandas", "numpy", "data analysis", "sql", "excel", "data visualisation",
                  "statistics" ]
    })

    profiles.append({
        "name": "Machine Learning Engineer",
        "skills": ["python", "machine learning", "tensorflow", "pytorch", "scikit-learn", "deep learning",
                  "model training", "nlp", "computer vision" ]
    })

    profiles.append({
        "name": "Data Scientist",
        "skills": ["machine learninh", "statistics", "pandas", "numpy", "data modeling", "experimentation",
                  "data visualisation"  ]
    })

    profiles.append({
        "name": "Cloud Engineer",
        "skills": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "linux", "infrastructure" ]
    })

    profiles.append({
        "name": "Frontend Developer",
        "skills": ["html", "css", "javascript", "React", "Tailwind", "Bootstrap", "Typescript", "Angulat", "svelte",
                   "figma", "ui/ux" ]
    })


    for profile in profiles:
        profile["vector"] = embed.encode(profile["skills"])

    return profiles


def classify_profile(profile_vector):
    """
    Classifie le profil en comparant son vecteur d'embedding avec les profils de référence.
    Utilise la similarité cosinus pour trouver le profil de référence le plus proche.
    
    Args:
        profile_vector: Le vecteur d'embedding calculé à partir des compétences du candidat
    
    Returns:
        str: Le type de profil le plus similaire (ex: 'Backend Developer')
    """
    best_match = None
    best_similarity = 0
    profiles = reference_profiles()
    for reference_profile in profiles:
        similarity = util.cos_sim(profile_vector, reference_profile["vector"])[0][0].item()

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = reference_profile["name"]

    return best_match



def skill_category(skill):
    """
    Classe une compétence dans une catégorie prédéfinie.
    Catégories disponibles: backend, ml (machine learning), data, cloud, infra (infrastructure), other.
    
    Args:
        skill (str): La compétence à classer
    
    Returns:
        str: La catégorie de la compétence ('backend', 'ml', 'data', 'cloud', 'infra', ou 'other')
    """
    backend_skills = [
        "django", "fastapi", "flask", "api", "rest", "graphql"
    ]

    frontend_skills = [
        "html", "css", "javascript", "React", "Tailwind", "Bootstrap", "Typescript", "Angulat", "svelte",
                   "figma", "ui/ux"
    ]

    ml_skills = [
        "machine learning", "tensorflow", "pytorch",
        "deep learning", "nlp", "computer vision"
    ]

    data_skills = [
        "pandas", "numpy", "data analysis",
        "data visualization", "statistics"
    ]

    cloud_skills = [
        "aws", "azure", "gcp", "cloud"
    ]

    infra_skills = [
        "docker", "kubernetes", "ci/cd", "microservices"
    ]


    if skill in backend_skills:
        return "backend"
    
    if skill in frontend_skills:
        return "frontend"

    if skill in ml_skills:
        return "ml"

    if skill in data_skills:
        return "data"

    if skill in cloud_skills:
        return "cloud"

    if skill in infra_skills:
        return "infra"

    return "other"


def analyze_axes(skills):
    """
    Analyse la distribution des compétences par catégorie (axes).
    Compte le nombre de compétences dans chaque catégorie pour tracer le profil technologique.
    
    Args:
        skills (list): Liste des compétences du candidat
    
    Returns:
        dict: Dictionnaire avec les catégories comme clés et le nombre de compétences comme valeurs
    """
    axes = {
        "backend": 0,
        "frontend": 0,
        "data": 0,
        "ml": 0,
        "cloud": 0,
        "infra": 0,
        "other": 0
    }

    for skill in skills:
        category = skill_category(skill)
        axes[category] += 1

    return axes



def profile_from_axes(axes):

    if axes["ml"] >= 3:
        return "Machine Learning Engineer"
    
    if axes["data"] >= 3:
        return "Data Analyst"
    
    if axes["backend"] >= 3:
        return "Backend Developer"
    
    if axes["frontend"] >= 3:
        return "Frontend Developer"
    
    if axes["cloud"] >= 3:
        return "Cloud Engineer"
    
    return "Hybrid / Generalist"


def base_score(profile_type):
    """
    Attribue un score de base selon le type de profil.
    Le score de base dépend de la difficulté et de la demande du profil en marché.
    
    Args:
        profile_type (str): Le type de profil (ex: 'Backend Developer', 'Machine Learning Engineer')
    
    Returns:
        int: Le score de base pour ce type de profil (entre 35 et 55)
    """
    if profile_type == "Backend Developer":
        return 40
    
    if profile_type == "Frontend Developer":
        return 38

    if profile_type == "Fullstack Developer":
        return 45

    if profile_type == "Data Analyst":
        return 42

    if profile_type == "Machine Learning Engineer":
        return 55

    if profile_type == "Data Scientist":
        return 50

    if profile_type == "Cloud Engineer":
        return 50

    return 35 


def weight_level(level):
    """
    Applique un multiplicateur de poids selon le niveau d'expérience estimé.
    Junior: -15 points, Intermédiaire: 0 points, Avancé: +20 points.
    
    Args:
        level (str): Le niveau d'expérience ('junior', 'intermediate', 'advanced')
    
    Returns:
        int: L'ajustement de score en fonction du niveau
    """
    if level == "junior":
        return -15

    if level == "intermediate":
        return 0

    if level == "advanced":
        return +20

    return 0


def contains_advanced_tools(skills):
    """
    Détecte si le candidat possède des compétences avancées dans les domaines spécialisés.
    Retourne True si au moins 2 compétences appartiennent à ml, infra ou cloud.
    Utilé pour estimer le niveau d'expérience.
    
    Args:
        skills (list): Liste des compétences du candidat
    
    Returns:
        bool: True si le candidat a des outils/compétences avancés, False sinon
    """
    category_counts = {}

    for skill in skills:
        category = skill_category(skill)
        category_counts[category] = category_counts.get(category, 0) + 1

    for category in category_counts:
        if category_counts[category] >= 2 and category in ["ml", "infra", "cloud"]:
            return True

    return False



def bonus_coherent_stack(skills):
    """
    Calcule un bonus de points si les compétences forment un "stack" cohérent.
    Stack cohérent = ensemble harmonieux de technologies qui fonctionnent bien ensemble.
    - Backend: +10 points si ≥3 compétences
    - ML: +10 points si ≥2 compétences
    - Cloud: +8 points si ≥2 compétences
    
    Args:
        skills (list): Liste des compétences du candidat
    
    Returns:
        int: Points bonus pour cohérence technologique
    """
    category_counts = {}
    for skill in skills:
        category = skill_category(skill)
        category_counts[category] = category_counts.get(category, 0) + 1

    bonus = 0

    if category_counts.get("backend", 0) >= 3:
        bonus += 10

    if category_counts.get("ml", 0) >= 2:
        bonus += 10

    if category_counts.get("cloud", 0) >= 2:
        bonus += 8

    return bonus



def estimate_level(skills):
    """
    Estime le niveau d'expérience du candidat basé sur le nombre et le type de compétences.
    - Junior: moins de 3 compétences
    - Avancé: possède des outils avancés (ml, infra, cloud)
    - Intermédiaire: entre junior et avancé
    
    Args:
        skills (list): Liste des compétences du candidat
    
    Returns:
        str: Le niveau estimé ('junior', 'intermediate', 'advanced')
    """
    if len(skills) < 3:
        return "junior"

    if contains_advanced_tools(skills):
        return "advanced"

    return "intermediate"


def compute_score(skills, profile_type, level):
    """
    Calcule le score global du candidat en combinant plusieurs facteurs:
    1. Score de base selon le type de profil
    2. Ajustement selon le niveau d'expérience
    3. Bonus pour cohérence et harmonie du stack technologique
    
    Args:
        skills (list): Liste des compétences du candidat
        profile_type (str): Le type de profil déterminé
        level (str): Le niveau d'expérience estimé
    
    Returns:
        int: Le score global (entre ~20 et ~85)
    """
    score = base_score(profile_type)

    score += weight_level(level)

    score += bonus_coherent_stack(skills)

    return score



def analyze_profile(text):
    """
    Analyse complète d'un profil à partir d'un texte (ex: CV).
    Effectue l'extraction des compétences, la classification du profil, l'estimation du niveau,
    et calcule un score global.
    
    Args:
        text (str): Le texte à analyser (CV, description professionnelle, etc.)
    
    Returns:
        dict: Dictionnaire contenant:
            - summary: Résumé du texte
            - skills: Liste des compétences extraites
            - profile_type: Type de profil détecté
            - level: Niveau d'expérience estimé
            - axes: Distribution des compétences par catégorie
            - score: Score global du profil
    """
    skills = extract(text)
    axes = analyze_axes(skills)

    profile_type = profile_from_axes(axes)

    if profile_type == "Hybrid / Generalist":

        profile_vector = embed.encode(skills)
        profile_type = classify_profile(profile_vector)
    
    level = estimate_level(skills)
    
    summary = build_summary(axes, profile_type, level)
    score = compute_score(skills, profile_type, level)

    return {
        "summary": summary,
        "skills": skills,
        "profile_type": profile_type,
        "level": level,
        "axes": axes,
        "score": score
    }


