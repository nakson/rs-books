# Book Recommendation 

### How to Run

1.  Install environment

    ```shell
    cd ./src
    pip install -r requirements.txt
    ```

2.  Run the code

    ```shell
    uvicorn main:app
    ```

3.  Open webpage from Chrome

    - [Web Server for Chrome](https://chrome.google.com/webstore/detail/web-server-for-chrome/ofhbbkphhbklhfoeikjpcbhemlocgigb?hl=en)

      <img src="https://s2.loli.net/2022/04/02/Th4YaJHfQyN9URx.png" alt="image-2022040230445665 PM" style="zoom:33%; margin:0;" />

### Book DataSet

- [Dowload link - Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip)

- [more](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

  <img src="https://s2.loli.net/2022/04/01/o7hCK6EQkJXrl4i.png" alt="image-2022040195223150 PM" style="zoom: 50%;" />

### Screenshots of All User Interfaces

![img](https://s2.loli.net/2022/05/31/s3c2jCbVOlTLJYo.png)Figure 1: When users open the website, there is a pop-up window to get the user’s name and age.

![img](https://s2.loli.net/2022/05/31/rMqsaRoTklxYF4f.png)Figure 2: The age drop-down bar provides 5 age options.

![img](https://s2.loli.net/2022/05/31/a9hkMlKpeID5Rqd.png)Figure 3: This is the entry of our method 1 system, showing a welcome message and the books filtered by the user-selected age.

![img](https://s2.loli.net/2022/05/31/HOaFiVmRKYAkhrD.jpg)Figure 4: After the user clicks a like button for a book, the algorithm will suggest five similar books.

![img](https://s2.loli.net/2022/05/31/hQpDCEJmous2e9d.png)Figure 5: The user can like or dislike the recommendation and submit the result through the button.

![img](https://s2.loli.net/2022/05/31/KXZAv2Fdxjy9wUD.png)Figure 6: The recommendation list will be updated based on the user's like or dislike, and refreshed. The user can constantly refresh by clicking the refresh button.

![img](https://s2.loli.net/2022/05/31/ZmcskiDwMVEjoJp.png)Figure 7: This is the entry of our method 2, showing a welcome message and books in the user’s selected age group. This method elicits users’ explicit preferences by ratings.

![img](https://s2.loli.net/2022/05/31/QYmUiVzGLg3aI4F.png)Figure 8: The algorithm will take users' ratings as input, and output recommended books, here we only randomly display 6 books. The user can click the refresh button to refresh the books.

![img](https://s2.loli.net/2022/05/31/IQOPKw7dliXkmyS.png)Figure 9: Users can rate the recommended books through face rating.

![img](https://s2.loli.net/2022/05/31/BZwhS5o39U7rDKm.png)Figure 10: The submit button will remove the recommended items whose face rating is smaller than 3, and refresh the books.
