## Programming with Data (ENDG233) Final Project

### Generation of the Dataset

The aim of this project was to import, process, and analyze a dataset.
The dataset I chose to use was a collection of my own and my peer's historical music listening data (scrobbles).
This data has been tracked over the last ~6 years on [Last.FM](https://www.last.fm/user/jbreidfjord) and was collected via the [Last.FM API](https://www.last.fm/api).
The data was then cleaned and stored into 3 csv files, each containing track data, album data, or artist data.

### Data Analysis of Personal Music Data

For the analysis portion of the project, the goal was to provide a menu of analysis options for the user to choose from, then process, output, and plot the data accordingly.

The analysis options I implemented are as follows:

- Mainstream Score:
  - A weighted average of the total number of global plays for artists
  - Weights are calculated as the selected user's scrobbles for a given artist over the sum of all of the user's scrobbles (i.e. what percentage of your listening was to that artist)
  - Each artist then has their contribution to the mainstream score calculated
  - Plot is a pie chart with the top 9 contributing artists + the sum of all others
  - Output contains the selected user's most mainstream and most obscure artist, with each of their contributions
- Taste Similarity:
  - Calculates the correlation between two selected user's scrobbles for any of tracks, albums, or artists
  - Uses the Pearson correlation coefficient
  - Plot is a scatter plot with one user providing x values and the other providing y values
  - The line y=x is plotted and for any given point, the distance from that line relates how correlated that point is, where points on the line are perfectly correlated
  - Most correlated and least correlated points are outputted and highlighted on the plot
- Discography Depth:
  - Calculates the number of unique albums scrobbled by the selected user for all artists
  - Plot is a bar graph displaying the top 10 deepest discography dives (most albums scrobbled), with a line plotted to display the mean
  - Deepest dive and mean value are outputted
- Average Track Duration:
  - A weighted average of the selected user's song lengths in seconds
  - Similar to the mainstream score, weights are calculated as a percentage of the user's total scrobbles

### Planning and Timeline

Planning for the project was done in the form of a to-do list containing possible ideas and functionality or specific items to be implemented or completed.
The timeline for project milestones was tracked in the form of the commit history to the git repo.

Noteable milestone commits include:

- [Finalize data collection](https://github.com/JBreidfjord/endg233-final-project/commit/5086231c175a8967639ef47946907d6037afe72b) (Dec 2, 2021)
- [Implement User class and mainstream functions](https://github.com/JBreidfjord/endg233-final-project/commit/d51a50b3f5db94796269fa1b136790bd70a25a8e) (Dec 3, 2021)
- [Define discography depth and add color to plots](https://github.com/JBreidfjord/endg233-final-project/commit/657ecf88f024adc77cf040f798f88e40f93aadde) (Dec 3, 2021)
- [Fix data error](https://github.com/JBreidfjord/endg233-final-project/commit/d5385610fbe979a9b64b32a20c36553907d125cc) (Dec 4, 2021)
- [Define similarity and plot_similarity functions](https://github.com/JBreidfjord/endg233-final-project/commit/85db6f01bb62e6e5912246023ecd1acd864acb4e) (Dec 4, 2021)
- [Define menu function for user input](https://github.com/JBreidfjord/endg233-final-project/commit/3ce0a99cce4653f69b9559b7095d9402a96366ea) (Dec 6, 2021)
