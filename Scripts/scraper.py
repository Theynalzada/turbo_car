# Importing Dependencies
import pandas as pd
import requests
import warnings
import logging
import yaml
import bs4
import os

# Filtering potential warnings
warnings.filterwarnings(action = "ignore")

# Creating a logger file
logging.basicConfig(filename = "/Users/kzeynalzade/Documents/Turbo Project/Logs/web_scraping.log", 
                    filemode = "w", 
                    format = "%(asctime)s - %(levelname)s - %(message)s", 
                    level = logging.INFO)

# Loading the configuration file
with open(file = "/Users/kzeynalzade/Documents/Turbo Project/Configuration/config.yml") as yaml_file:
    config = yaml.safe_load(stream = yaml_file)

# Extracting the target url
target_url = config.get("target_url")

# Extracting the base url
base_url = config.get("base_url")

# Defining a function to scrape data from the target url
def scrape_data(target_url = None, verify_certificate = True, parser = "html.parser"):
    """
    This function is used to scrape data from https://www.turbo.az website.
    
    Args:
        verify_certificate: Whether or not to verify the certificate.
        parser: A type of the parser to scrape data.
        
    Returns:
        Writes scraped data to a csv file.
    """
    # Creating an empty list to store the data frames
    data_frames = []

    # Sending a request to the target url
    response = requests.get(url = target_url, verify = verify_certificate)

    # Asserting the status code to be equal to 200
    assert response.status_code == 200

    # Instantiating the soup
    soup = bs4.BeautifulSoup(markup = response.content, features = parser)

    # Creating a list of car elements
    cars = soup.find_all(name = "div", attrs = {"class": "products-i"})

    # Creating a while loop
    while True:
        # Looping through each car element
        for car in cars:
            # Extracting the href
            href = car.find(name = "a", attrs = {"class": "products-i__link"}).get("href")

            # Creating the link for the car
            link = f"{base_url}{href}"

            # Extracting the car price
            price = car.find(name = "div", attrs = {"class": "product-price"}).get_text().strip().replace(" ", "")

            # Removing the AZN currency from the price
            price = price.replace("AZN", "")

            # Removing the Euro icon from the price
            price = price.replace("€", "")

            # Removing the Dollar icon from the price
            price = price.replace("$", "")

            # Extracting the currency
            currency = car.find(name = "div", attrs = {"class": "product-price"}).span.get_text().strip()

            # Creating a condition based on the currency
            if currency == "€":
                # Replacing the icon with a string in case it is a Euro icon
                currency = "EUR"
            elif currency == "$":
                # Replacing the icon with a string in case it is a Dollar icon
                currency = "USD"
            else:
                # Passing in case the currency is AZN
                pass

            # Creating a list of year, engine, mileage
            car_attributes = car.find(name = "div", attrs = {"class": "products-i__attributes products-i__bottom-text"}).get_text().strip().split(", ")

            # Unpacking the list
            year, engine, mileage = car_attributes
            
            # Removing the extra spaces in mileage
            mileage = mileage.replace(" ", "").replace("km", " km")

            # Extracting the element for the availability of exchange
            exchange_element = car.find(name = "div", attrs = {"class": "products-i__icon products-i__icon--barter"})

            # Creating a condition based on the availability of exchange
            if exchange_element is not None:
                # Extracting the string
                exchange = exchange_element.get_text().strip()
            else:
                # Assigning a different string in case the condition is not satisfied
                exchange = "Barter mümkün deyil"

            # Extracting the loan element
            loan_element = car.find(name = "div", attrs = {"class": "products-i__icon products-i__icon--loan"})

            # Creating a condition based on the loan
            if loan_element is not None:
                # Extracting the string
                loan = loan_element.get_text().strip()
            else:
                # Assigning a different string in case the condition is not satisfied
                loan = "Kreditde deyil"

            # Extracting the saloon element
            saloon_car = car.find(name = "div", attrs = {"class": "products-i__label products-i__label--salon"})

            # Creating a condition based on the saloon
            if saloon_car is not None:
                # Extracting the string
                saloon = saloon_car.get_text().strip()
            else:
                # Assigning a different string in case the condition is not satisfied
                saloon = "Şəxsi"

            # Sending a new request to the car link
            car_response = requests.get(url = link, verify = verify_certificate)

            # Creating a condition based on status code
            if car_response.status_code == 200:
                # Asserting the status code to be equal to 200
                assert car_response.status_code == 200

                # Passing in case the condition is satisfied
                pass
            else:
                # Breaking the loop
                break

            # Instantiating the soup for the particular car
            car_soup = bs4.BeautifulSoup(markup = car_response.content, features = parser)

            # Extracting the description
            description_element = car_soup.h2

            # Creating a condition based on the description element
            if description_element is not None:
                # Extracting the description
                description = description_element.get_text().strip()
            else:
                # Assigning a different string
                description = "Satıcı tərəfindən əlavə bir qeyd yoxdur"

            # Creating a list of elements small car images
            car_image_elements_s = car_soup.find(name = "div", attrs = {"class": "product-photos-thumbnails_s"})

            # Creating a list of elements medium car images
            car_image_elements_m = car_soup.find(name = "div", attrs = {"class": "product-photos-thumbnails_m"})

            # Creating a list of elements large car images
            car_image_elements_l = car_soup.find(name = "div", attrs = {"class": "product-photos-thumbnails_l"})

            # Creating a list of elements large car images
            car_image_elements_xl = car_soup.find(name = "div", attrs = {"class": "product-photos-thumbnails_xl"})

            # Creating a condition based on the car image elements
            if car_image_elements_s is not None:
                # Creating a list of small image elements
                car_image_elements = car_image_elements_s.findChildren()
            elif car_image_elements_m is not None:
                # Creating a list of medium image elements
                car_image_elements = car_image_elements_m.findChildren()
            elif car_image_elements_l is not None:
                # Creating a list of large image elements
                car_image_elements = car_image_elements_l.findChildren()
            else:
                # Creating a list of extra large image elements
                car_image_elements = car_image_elements_xl.findChildren()

            # Looping through each element
            car_images = [car_image.get("src") for car_image in car_image_elements if car_image.get("src") is not None]

            # Creating a seller element
            seller_element = car_soup.find(name = "div", attrs = {"class": "seller-name"})

            # Creating a condition based on the seller element
            if seller_element is not None:
                # Extracting the seller name
                seller = seller_element.get_text().strip()
            else:
                # Creating an element based on the current status of the deal
                status_element = car_soup.find(name = "div", attrs = {"class": "status-message status-message--expired"})

                # Creating a condition based on the current status of the deal
                if status_element is None:
                    # Extracting the name of the saloon
                    seller = car_soup.h1.get_text().strip()
                else:
                    # Extracting the string in case the condition is satisfied
                    seller = status_element.get_text().strip()
                    
            # Creating an element for the car shop location
            car_shop_element = car_soup.find(name = "a", attrs = {"class": "shop--location"})
            
            # Creating a condition based on the car shop element
            if car_shop_element is not None:
                # Extracting a car shop description
                shop_location = car_shop_element.get_text().strip()

                # Extracting the link for the Google Maps
                google_maps_link = car_shop_element.get("href")
                
                # Extracting the number of cars on sale for a particular car shop
                n_cars_by_shop = car_soup.find(name = "a", attrs = {"class": "shop--products-count"}).get_text().strip()
            else:
                # Assigning a different string
                shop_location = "Şəxsi avtomobil satışlarında adres qeyd olunmur"

                # Assigning a different string
                google_maps_link = "Şəxsi avtomobil satışlarında Google Map üzərindən adres link qeyd olunmur"
                
                # Assigning a different string
                n_cars_by_shop = "Şəxsi avtomobil satışlarında satıcı tərəfindən digər elanlar qeyd olunmur"

            # Extracting the number of times it has been seen
            n_seen = int(car_soup.find(name = "div", attrs = {"class": "product-statistics"}).p.get_text().split(": ")[-1])

            # Extracting the last updated date
            last_updated_at = car_soup.find(name = "div", attrs = {"class": "product-statistics"}).p.find_next_sibling().get_text().split(": ")[-1]

            # Extracting the deal ID
            ID = car_soup.find(name = "div", attrs = {"class": "product-statistics"}).p.find_next_siblings()[-1].get_text().split(": ")[-1]

            # Creating a list of car elements
            car_elements = [element.div.get_text().strip() for element in car_soup.find_all(name = "li", attrs = {"class": "product-properties-i"})]

            # Extracting the brand of the car
            car_brand = car_elements[1]

            # Extracting the model of the car
            car_model = car_elements[2]

            # Extracting the type of the car
            car_type = car_elements[4].replace(" ", "")

            # Extracting the color of the car
            color = car_elements[5]
            
            # Extracting the horse power of the car
            hp = car_elements[7]

            # Extracting the fuel type of the car
            fuel_type = car_elements[8]

            # Extracting the speed box of the car
            speed_box = car_elements[10]

            # Extracting the transmission of the car
            transmission = car_elements[11]

            # Extracting the condition of the car
            condition = car_elements[12]

            # Creating a list of the functionalities of the car
            car_functionalities = [element.get_text().strip() for element in car_soup.find_all(name = "p", attrs = {"class": "product-extras-i"})]

            # Creating a seller element
            seller_phone_element = car_soup.find(name = "a", attrs = {"class": "phone"})

            # Creating a condition based on the seller element
            if seller_phone_element is not None:
                # Extracting the seller name
                seller_phone = seller_phone_element.get_text().strip()
            else:
                # Extracting the name of the saloon
                seller_phone = [element.get_text() for element in car_soup.find_all(name = "div", attrs = {"class": "shop-phones-i"})]

                # Creating a condition based on the number of phone numbers
                if len(seller_phone) == 1:
                    # Indexing the first value from the list
                    seller_phone = seller_phone[0]
                elif len(seller_phone) == 0:
                    # Assigning a different string in case the seller has not entered any contact number
                    seller_phone = "Əlaqə nömrəsi qeyd edilməyib"
                else:
                    # Passing in case the conditions are not satisfied
                    pass

            # Creating a dictionary to store the data
            data_dictionary = {"ID": f"ID-{ID}",
                               "vehicle_type": car_type,
                               "brand": car_brand,
                               "model": car_model,
                               "year": year,
                               "engine": engine,
                               "mileage": mileage,
                               "color": color,
                               "hp": hp,
                               "fuel_type": fuel_type,
                               "speed_box": speed_box,
                               "transmission": transmission,
                               "is_new": condition,
                               "exchange_available": exchange,
                               "is_saloon_car": saloon,
                               "loan_available": loan,
                               "functionalities": [car_functionalities],
                               "seller": seller,
                               "contact_info": [seller_phone],
                               "n_seen": n_seen,
                               "last_updated_at": last_updated_at,
                               "images": [car_images],
                               "link": link,
                               "description": description,
                               "shop_location": shop_location,
                               "google_maps_link": google_maps_link,
                               "n_cars_by_shop": n_cars_by_shop,
                               "currency": currency,
                               "price": price}

            # Storing the dictionary in a data frame
            data_frame = pd.DataFrame(data = data_dictionary, index = [0])

            # Appending the data frame to the list
            data_frames.append(data_frame)

        # Extracting the current page number
        current_page = soup.find(name = "span", attrs = {"class": "page current"}).get_text().strip()
        
        # Sending logs to the logger file
        logging.info(msg = f"Data from page {current_page} has been scraped")

        # Extracting the element for the next page
        next_page_element = soup.find(name = "span", attrs = {"class": "next"})

        # Creating a condition based on the next page
        if next_page_element is not None:
            # Extracting the href element for the next page
            next_page_href = next_page_element.a.get("href")

            # Updating the target url
            target_url = f"{base_url}{next_page_href}"

            # Sending a request to the target url
            response = requests.get(url = target_url, verify = verify_certificate)

            # Creating a condition based on the status code
            if response.status_code == 200:
                # Asserting the status code to be equal to 200
                assert response.status_code == 200

                # Passing in case the condition is satisfied
                pass
            else:
                # Breaking the loop
                break

            # Instantiating the soup
            soup = bs4.BeautifulSoup(markup = response.content, features = parser)

            # Creating a list of car elements
            cars = soup.find_all(name = "div", attrs = {"class": "products-i"})
        else:
            # Breaking the loop
            break

        # Concatenating the data frames vertically
        df = pd.concat(objs = data_frames, ignore_index = True)
        
        # Removing potential duplicate observations from the dataset
        df.drop_duplicates(subset = "ID", inplace = True, ignore_index = True)

        # Asserting the number of duplicate observations to be equal to zero
        assert df.duplicated(subset = "ID").sum() == 0
        
        # Shuffling the dataset
        df = df.sample(frac = 1.0, random_state = 42)
        
        # Reseting the index
        df.reset_index(drop = True, inplace = True)

        # Casting the data type of the last_updated_at variable
        df.last_updated_at = pd.to_datetime(arg = df.last_updated_at, dayfirst = True)
        
        # Creating a filepath
        filepath = "/Users/kzeynalzade/Documents/Turbo Project/Data/Raw Data/raw_data.csv"
        
        # Creating a condition based on the path
        if os.path.exists(path = filepath):
            # Reading the data from the csv file
            existing_df = pd.read_csv(filepath_or_buffer = filepath)
            
            # Creating a condition based on the data type of variable last_updated_at
            if existing_df.last_updated_at.dtype == "O":
                # Casting the data type of the last_updated_at variable
                existing_df.last_updated_at = pd.to_datetime(arg = existing_df.last_updated_at, yearfirst = True)
            else:
                # Passing in case the condition is not satisfied
                pass
            
            # Concatenating the scraped data with the existing data
            final_df = pd.concat(objs = [existing_df, df], ignore_index = True)
            
            # Removing potential duplicate observations from the dataset
            final_df.drop_duplicates(subset = "ID", inplace = True, ignore_index = True)
            
            # Asserting the number of duplicate observations to be equal to zero
            assert final_df.duplicated(subset = "ID").sum() == 0
            
            # Writing a data frame to a separate csv file
            final_df.to_csv(path_or_buf = filepath, index = False)
        else:
            # Writing a data frame to a separate csv file
            df.to_csv(path_or_buf = filepath, index = False)

# Running the script
if __name__ == "__main__":
    # Calling the function
    scrape_data(target_url = target_url)