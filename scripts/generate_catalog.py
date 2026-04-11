"""Generate a synthetic multilingual podcast catalog for demo purposes."""
# ruff: noqa: E501

import json
import random
from pathlib import Path

SEED = 42

# Episode duration distribution (seconds)
DURATION_MIN = 300  # 5 minutes
DURATION_MAX = 7200  # 2 hours
DURATION_MEAN = 1800  # 30 minutes
DURATION_STD = 1200  # 20 minutes

TOPICS = {
    "Technology": {
        "fr": [
            "Un podcast qui explore les dernières avancées en intelligence artificielle et leur impact sur notre quotidien. Des experts partagent leurs analyses sur les tendances technologiques.",
            "Chaque semaine, nous décryptons les innovations du monde numérique. De la cybersécurité au cloud computing, un tour d'horizon complet de la tech.",
            "Les nouvelles technologies transforment notre société. Ce podcast analyse les ruptures technologiques et leurs conséquences sur le monde du travail.",
        ],
        "en": [
            "A deep dive into artificial intelligence, machine learning, and the future of computing. Industry experts share insights on emerging tech trends.",
            "Weekly conversations about software engineering, open source, and developer culture. From startups to big tech, we cover the stories that matter.",
            "Exploring how technology shapes our world. From quantum computing to blockchain, we break down complex topics for curious minds.",
        ],
        "de": [
            "Ein Podcast über künstliche Intelligenz und die digitale Transformation. Experten diskutieren die neuesten Entwicklungen der Technologiebranche.",
            "Wöchentliche Gespräche über Software-Entwicklung und digitale Innovation. Von Start-ups bis Großkonzerne, wir berichten über die Technikwelt.",
        ],
    },
    "History": {
        "fr": [
            "Plongez dans les grandes époques de l'histoire de France. Des Gaulois à la Révolution, redécouvrez les événements qui ont façonné notre nation.",
            "Ce podcast raconte les histoires méconnues du passé. Guerres, découvertes, personnages oubliés : l'histoire comme vous ne l'avez jamais entendue.",
            "Une exploration des civilisations anciennes et de leur héritage. De l'Égypte antique à Rome, découvrez les fondements de notre monde moderne.",
        ],
        "en": [
            "Journey through the great events of world history. From ancient empires to modern revolutions, discover the stories that shaped civilization.",
            "Hidden stories from the past that changed the world. Forgotten heroes, lost civilizations, and pivotal moments retold for modern audiences.",
            "A weekly exploration of historical events and their lasting impact. Wars, discoveries, and cultural shifts that defined humanity.",
        ],
        "de": [
            "Eine Reise durch die großen Epochen der Weltgeschichte. Von antiken Imperien bis zu modernen Revolutionen.",
            "Verborgene Geschichten aus der Vergangenheit. Vergessene Helden und entscheidende Momente der Menschheitsgeschichte.",
        ],
    },
    "Science": {
        "fr": [
            "Les dernières découvertes scientifiques expliquées de manière accessible. Physique, biologie, astronomie : la science pour tous.",
            "Un podcast qui rend la science passionnante. Des chercheurs partagent leurs travaux et leurs découvertes avec enthousiasme.",
            "De la physique quantique à la génétique, explorez les frontières de la connaissance scientifique avec des experts passionnés.",
        ],
        "en": [
            "Breaking down the latest scientific discoveries. From particle physics to marine biology, making complex science accessible and exciting.",
            "Conversations with researchers pushing the boundaries of human knowledge. Astronomy, genetics, neuroscience, and more.",
            "A podcast about the wonders of the natural world. Climate science, evolution, and the mysteries of the universe explained.",
        ],
        "de": [
            "Die neuesten wissenschaftlichen Entdeckungen verständlich erklärt. Von Physik bis Biologie, Wissenschaft für alle.",
            "Gespräche mit Forschern an den Grenzen des Wissens. Astronomie, Genetik und Neurowissenschaften.",
        ],
    },
    "Sports": {
        "fr": [
            "L'actualité sportive décryptée par des experts. Football, rugby, tennis : analyses tactiques et coulisses du sport professionnel.",
            "Un podcast dédié aux exploits sportifs et aux athlètes qui repoussent les limites. Interviews exclusives et récits inspirants.",
        ],
        "en": [
            "In-depth analysis of the biggest stories in professional sports. Strategy, statistics, and behind-the-scenes insights from experts.",
            "Celebrating athletic achievement and the stories behind the scores. From underdogs to champions, the human side of sports.",
        ],
        "de": [
            "Sportanalysen und Hintergrundberichte. Fußball, Tennis und mehr: taktische Einblicke in den Profisport.",
        ],
    },
    "Culture": {
        "fr": [
            "Un regard curieux sur les arts et la culture contemporaine. Cinéma, littérature, musique : les créateurs d'aujourd'hui partagent leur vision.",
            "Explorez la richesse culturelle francophone. Théâtre, danse, arts visuels : un podcast pour les amoureux de la culture.",
        ],
        "en": [
            "Exploring contemporary arts and culture. Film, literature, music, and the creative minds shaping our cultural landscape.",
            "A podcast about the intersection of art and society. Museum exhibitions, literary trends, and cultural movements discussed.",
        ],
        "de": [
            "Ein Blick auf Kunst und zeitgenössische Kultur. Kino, Literatur, Musik: Kreative teilen ihre Vision.",
        ],
    },
    "Business": {
        "fr": [
            "Stratégies d'entreprise et entrepreneuriat. Des dirigeants partagent leurs expériences et conseils pour réussir dans le monde des affaires.",
            "L'économie décryptée pour les professionnels. Marchés financiers, management, innovation : les clés pour comprendre le business moderne.",
        ],
        "en": [
            "Business strategy and entrepreneurship insights. Leaders share their experiences building and scaling successful companies.",
            "Economics and finance explained for professionals. Market trends, management practices, and innovation strategies.",
        ],
        "de": [
            "Unternehmensstrategien und Entrepreneurship. Führungskräfte teilen ihre Erfahrungen und Tipps für geschäftlichen Erfolg.",
        ],
    },
    "Comedy": {
        "fr": [
            "Un podcast humoristique qui décortique l'actualité avec légèreté. Sketches, chroniques et fous rires garantis chaque semaine.",
            "Des humoristes se retrouvent pour des conversations hilarantes. Anecdotes, improvisations et moments de pure comédie.",
        ],
        "en": [
            "A comedy podcast that dissects current events with wit and humor. Sketches, rants, and guaranteed laughs every week.",
            "Comedians gather for hilarious conversations. Anecdotes, improvisation, and moments of pure comedic gold.",
        ],
        "de": [
            "Ein humorvoller Podcast, der aktuelle Themen mit Leichtigkeit behandelt. Sketche und Lacher jede Woche.",
        ],
    },
    "Politics": {
        "fr": [
            "Analyse politique approfondie et non partisane. Les enjeux nationaux et internationaux décryptés par des spécialistes.",
            "Comprendre la politique autrement. Débats, interviews et analyses pour éclairer les citoyens sur les décisions qui les concernent.",
        ],
        "en": [
            "Non-partisan political analysis and commentary. National and international issues explained by policy experts.",
            "Understanding politics through informed debate. Interviews with policymakers and analysis of the decisions that affect us all.",
        ],
        "de": [
            "Politische Analyse und Kommentare. Nationale und internationale Themen von Experten erklärt.",
        ],
    },
    "Health": {
        "fr": [
            "Santé et bien-être au quotidien. Des médecins et chercheurs partagent des conseils pratiques pour vivre mieux.",
            "Un podcast sur la santé mentale et physique. Nutrition, exercice, méditation : les clés d'une vie équilibrée.",
        ],
        "en": [
            "Daily health and wellness advice. Doctors and researchers share practical tips for living a healthier life.",
            "A podcast about mental and physical well-being. Nutrition, exercise, mindfulness, and the keys to a balanced life.",
        ],
        "de": [
            "Gesundheit und Wohlbefinden im Alltag. Ärzte und Forscher teilen praktische Tipps für ein besseres Leben.",
        ],
    },
    "Education": {
        "fr": [
            "Apprendre tout au long de la vie. Ce podcast explore les méthodes pédagogiques innovantes et les savoirs essentiels.",
            "Un podcast éducatif pour les curieux. Sciences, philosophie, langues : élargissez vos horizons chaque semaine.",
        ],
        "en": [
            "Lifelong learning made engaging. Exploring innovative teaching methods and essential knowledge for the modern world.",
            "An educational podcast for the curious. Science, philosophy, languages: expand your horizons every week.",
        ],
        "de": [
            "Lebenslanges Lernen leicht gemacht. Innovative Lehrmethoden und wesentliches Wissen für die moderne Welt.",
        ],
    },
    "Music": {
        "fr": [
            "Explorez l'univers musical dans toute sa diversité. Jazz, classique, électro, hip-hop : des artistes racontent leur parcours.",
            "L'histoire de la musique racontée à travers ses albums mythiques. Analyses, anecdotes et découvertes sonores.",
        ],
        "en": [
            "Exploring the world of music in all its diversity. Jazz, classical, electronic, hip-hop: artists share their journeys.",
            "The history of music told through its iconic albums. Analysis, stories, and sonic discoveries.",
        ],
        "de": [
            "Die Welt der Musik in all ihrer Vielfalt. Jazz, Klassik, Elektronik: Künstler erzählen ihre Geschichte.",
        ],
    },
    "Travel": {
        "fr": [
            "Partez à la découverte du monde depuis votre canapé. Récits de voyage, conseils pratiques et cultures du monde entier.",
            "Un podcast pour les voyageurs et les rêveurs. Destinations insolites, rencontres locales et aventures extraordinaires.",
        ],
        "en": [
            "Discover the world from your couch. Travel stories, practical tips, and cultures from around the globe.",
            "A podcast for travelers and dreamers. Unusual destinations, local encounters, and extraordinary adventures.",
        ],
        "de": [
            "Entdecken Sie die Welt von Ihrem Sofa aus. Reiseberichte, praktische Tipps und Kulturen aus aller Welt.",
        ],
    },
    "Crime": {
        "fr": [
            "Plongez dans les affaires criminelles les plus fascinantes. Enquêtes, témoignages et analyses de criminologues.",
            "True crime à la française. Les grandes affaires judiciaires racontées avec rigueur et humanité.",
        ],
        "en": [
            "Diving into the most fascinating criminal cases. Investigations, testimonies, and expert criminological analysis.",
            "True crime stories told with rigor and humanity. Cold cases, investigations, and the pursuit of justice.",
        ],
        "de": [
            "Die faszinierendsten Kriminalfälle. Ermittlungen, Zeugenaussagen und kriminologische Analysen.",
        ],
    },
    "Environment": {
        "fr": [
            "Écologie et développement durable. Comprendre les enjeux environnementaux et découvrir les solutions de demain.",
            "Un podcast sur la transition écologique. Biodiversité, climat, énergie : agir pour la planète au quotidien.",
        ],
        "en": [
            "Ecology and sustainability. Understanding environmental challenges and discovering tomorrow's solutions.",
            "A podcast about the ecological transition. Biodiversity, climate, energy: acting for the planet every day.",
        ],
        "de": [
            "Ökologie und Nachhaltigkeit. Umweltherausforderungen verstehen und Lösungen für morgen entdecken.",
        ],
    },
    "Philosophy": {
        "fr": [
            "Les grandes questions philosophiques accessibles à tous. Liberté, justice, bonheur : réfléchir pour mieux vivre.",
            "Un podcast qui fait penser. Philosophes classiques et contemporains éclairent notre compréhension du monde.",
        ],
        "en": [
            "Big philosophical questions made accessible. Freedom, justice, happiness: thinking to live better.",
            "A podcast that makes you think. Classic and contemporary philosophers illuminate our understanding of the world.",
        ],
        "de": [
            "Große philosophische Fragen für alle zugänglich gemacht. Freiheit, Gerechtigkeit, Glück: Denken für ein besseres Leben.",
        ],
    },
}

TITLE_TEMPLATES = {
    "fr": [
        "Parlons {topic}",
        "{topic} en question",
        "Le monde de {topic_lower}",
        "Décryptage {topic}",
        "{topic} aujourd'hui",
        "Au coeur de {topic_lower}",
        "L'heure de {topic_lower}",
        "Regards sur {topic_lower}",
        "{topic} sans filtre",
        "Le podcast {topic}",
    ],
    "en": [
        "The {topic} Hour",
        "{topic} Unpacked",
        "Inside {topic}",
        "{topic} Today",
        "The {topic} Podcast",
        "Beyond {topic}",
        "{topic} Explained",
        "Deep Dive: {topic}",
        "{topic} Weekly",
        "All About {topic}",
    ],
    "de": [
        "{topic} heute",
        "{topic} im Fokus",
        "Die {topic}-Stunde",
        "{topic} erklärt",
        "Alles über {topic}",
    ],
}

TOPIC_TRANSLATIONS = {
    "Technology": {"fr": "Technologie", "en": "Technology", "de": "Technologie"},
    "History": {"fr": "Histoire", "en": "History", "de": "Geschichte"},
    "Science": {"fr": "Science", "en": "Science", "de": "Wissenschaft"},
    "Sports": {"fr": "Sport", "en": "Sports", "de": "Sport"},
    "Culture": {"fr": "Culture", "en": "Culture", "de": "Kultur"},
    "Business": {"fr": "Business", "en": "Business", "de": "Wirtschaft"},
    "Comedy": {"fr": "Humour", "en": "Comedy", "de": "Comedy"},
    "Politics": {"fr": "Politique", "en": "Politics", "de": "Politik"},
    "Health": {"fr": "Santé", "en": "Health", "de": "Gesundheit"},
    "Education": {"fr": "Éducation", "en": "Education", "de": "Bildung"},
    "Music": {"fr": "Musique", "en": "Music", "de": "Musik"},
    "Travel": {"fr": "Voyage", "en": "Travel", "de": "Reisen"},
    "Crime": {"fr": "Crime", "en": "Crime", "de": "Kriminalität"},
    "Environment": {"fr": "Environnement", "en": "Environment", "de": "Umwelt"},
    "Philosophy": {"fr": "Philosophie", "en": "Philosophy", "de": "Philosophie"},
}

EPISODE_TEMPLATES = {
    "fr": ["Épisode {n}", "Chapitre {n}", "Partie {n}", "#{n}"],
    "en": ["Episode {n}", "Chapter {n}", "Part {n}", "#{n}"],
    "de": ["Folge {n}", "Kapitel {n}", "Teil {n}", "#{n}"],
}


def generate_catalog(seed: int = SEED) -> dict:
    """Generate a synthetic multilingual podcast catalog."""
    rng = random.Random(seed)
    programs = []
    program_counter = 0
    media_counter = 0

    lang_targets = {"fr": 100, "en": 70, "de": 30}

    for lang, target_count in lang_targets.items():
        topics_list = list(TOPICS.keys())
        generated = 0

        while generated < target_count:
            topic = topics_list[generated % len(topics_list)]
            descriptions = TOPICS[topic][lang]
            description = rng.choice(descriptions)

            topic_local = TOPIC_TRANSLATIONS[topic][lang]
            title_tpl = rng.choice(TITLE_TEMPLATES[lang])
            title = title_tpl.format(topic=topic_local, topic_lower=topic_local.lower())
            if generated >= len(topics_list):
                title += f" #{generated // len(topics_list) + 1}"

            program_id = f"prg_{program_counter:04d}"
            program_counter += 1

            num_episodes = rng.randint(3, 8)
            media = []
            ep_template = rng.choice(EPISODE_TEMPLATES[lang])
            for ep in range(1, num_episodes + 1):
                media_id = f"med_{media_counter:05d}"
                media_counter += 1
                duration = max(
                    DURATION_MIN, min(DURATION_MAX, int(rng.gauss(DURATION_MEAN, DURATION_STD)))
                )
                media.append(
                    {
                        "media_id": media_id,
                        "episode": ep,
                        "duration": duration,
                        "title": ep_template.format(n=ep),
                    }
                )

            programs.append(
                {
                    "program_id": program_id,
                    "title": title,
                    "description": description,
                    "lang": lang,
                    "media": media,
                }
            )
            generated += 1

    return {"programs": programs}


if __name__ == "__main__":
    catalog = generate_catalog()
    output_path = Path(__file__).parent.parent / "data" / "catalog.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    langs = {}
    total_media = 0
    for p in catalog["programs"]:
        langs[p["lang"]] = langs.get(p["lang"], 0) + 1
        total_media += len(p["media"])
    print(f"Generated {len(catalog['programs'])} programs, {total_media} episodes")
    print(f"By language: {langs}")
