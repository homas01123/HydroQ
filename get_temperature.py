import requests
import argparse
import datetime

def get_surface_temperature(lat, lon, date_str):
    """
    Fetch surface temperature for a specific lat, lon, and date using NASA POWER API.
    
    Parameters:
    lat (float): Latitude (-90 to 90)
    lon (float): Longitude (-180 to 180)
    date_str (str): Date in format YYYY-MM-DD
    
    Returns:
    dict: Temperature data
    """
    # Parse the date string
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start_date = date_obj.strftime("%Y%m%d")
    end_date = date_obj.strftime("%Y%m%d")
    
    # NASA POWER API endpoint
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Define parameters
    params = {
        "parameters": "T2M,TS",  # T2M: Temperature at 2 Meters, TS: Earth Skin Temperature
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract temperature values
        try:
            properties = data['properties']
            parameter = properties['parameter']
            
            # Get the date key in the format expected by the API response
            date_key = date_obj.strftime("%Y%m%d")
            
            # Get temperatures
            t2m = parameter['T2M'][date_key] if 'T2M' in parameter else None
            ts = parameter['TS'][date_key] if 'TS' in parameter else None
            
            return {
                'air_temperature_2m': t2m,
                'skin_temperature': ts
            }
        except (KeyError, TypeError) as e:
            print(f"Error parsing response: {e}")
            return None
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch surface temperature data for a specific location and date.')
    parser.add_argument('lat', type=float, help='Latitude (-90 to 90)')
    parser.add_argument('lon', type=float, help='Longitude (-180 to 180)')
    parser.add_argument('date', help='Date in format YYYY-MM-DD')
    
    args = parser.parse_args()
    
    # Get temperature data
    temp_data = get_surface_temperature(args.lat, args.lon, args.date)
    
    if temp_data:
        print("\nSurface Temperature Data:")
        print(f"Location: {args.lat}, {args.lon}")
        print(f"Date: {args.date}")
        
        if temp_data['air_temperature_2m'] is not None:
            print(f"Air Temperature at 2m: {temp_data['air_temperature_2m']} °C")
        
        if temp_data['skin_temperature'] is not None:
            print(f"Surface Skin Temperature: {temp_data['skin_temperature']} °C")
            print("(This is equivalent to LST for land areas or SST for water bodies)")
    else:
        print("Failed to retrieve temperature data.")

if __name__ == "__main__":
    main()