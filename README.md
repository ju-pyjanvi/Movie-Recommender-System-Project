# Movie Recommender System Project

A Streamlit-based movie recommendation application that suggests movies based on similarity metrics and content-based filtering.

## Overview

This project implements an intelligent movie recommendation system that analyzes movie attributes and finds similar movies based on various factors including genres, ratings, release years, and more. The system provides an interactive web interface for users to discover new movies.

## Live Application

Try the app here: https://movie-recommender-project120.streamlit.app/

## Features

- 🎬 **Interactive Streamlit App** - User-friendly interface for movie discovery
- 🖼️ **Poster Images** - Visual representation of recommended movies
- ⭐ **IMDb Ratings** - View ratings for recommended movies
- 🎭 **Genre Information** - See genre classifications for each movie
- 📅 **Release Years** - Learn when each movie was released
- 🔍 **Content-Based Filtering** - Similarity-based recommendations using multiple movie attributes

## Project Structure

```
├── app.py                    # Main Streamlit application
├── main.py                   # Core application logic
├── main.ipynb                # Jupyter notebook with full workflow
├── movies.pkl                # Pre-trained movie data (pickled)
├── tmdb_5000_credits.csv     # TMDB credits dataset
├── tmdb_5000_movies.csv      # TMDB movies dataset
└── requirements.txt          # Python dependencies
```

## Technology Stack

- **Streamlit** - Web application framework
- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms for similarity metrics
- **Jupyter Notebook** - Development and experimentation
- **The Movie Database (TMDB)** - Data source

## How It Works

The recommendation system uses a **content-based filtering approach**:

1. **Data Loading** - Loads movie data from TMDB datasets
2. **Feature Engineering** - Extracts relevant features (genres, credits, ratings, etc.)
3. **Similarity Calculation** - Computes similarity between movies using cosine similarity
4. **Recommendation Generation** - Finds the most similar movies to a user's selected movie

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ju-pyjanvi/Movie-Recommender-System-Project.git
cd Movie-Recommender-System-Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Datasets

This project uses the TMDB 5000 movie dataset which includes:

- **tmdb_5000_movies.csv** - Contains movie information (title, genres, overview, ratings, etc.)
- **tmdb_5000_credits.csv** - Contains cast and crew information for each movie

### Dataset Features

- Movie titles and overviews
- Genre classifications
- IMDb ratings and vote counts
- Release dates
- Cast and crew information
- Runtime and budget data
- Production companies and countries

## Usage

1. Launch the Streamlit app
2. Select or search for a movie you like
3. View recommended movies based on similarity metrics
4. Explore movie posters, ratings, genres, and release information

## Repository Statistics

- **Language:** 90.2% Jupyter Notebook, 9.8% Python
- **Commits:** 26
- **Stars:** 1
- **Status:** Active

## Author

Created by [ju-pyjanvi](https://github.com/ju-pyjanvi)

## License

Please check the repository for license information.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## Acknowledgments

- TMDB (The Movie Database) for providing the movie data
- Streamlit for the web framework
- The open-source community for various libraries used in this project

## Future Enhancements

- Collaborative filtering based on user ratings
- Machine learning model improvements
- User history tracking
- Personalized recommendations
- Enhanced UI/UX features
- API integration for real-time data updates
