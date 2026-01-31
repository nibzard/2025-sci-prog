'use client';

import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet';
import { Icon } from 'leaflet';
import { useEffect } from 'react';
import type { CityData } from '@/lib/types';
import 'leaflet/dist/leaflet.css';

interface MapProps {
  cities: CityData[];
  onCitySelect: (city: string) => void;
  selectedCity: string | null;
  propertyType: 'apartments' | 'houses';
}

// Create pin icons
const defaultIcon = new Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3b82f6" width="32" height="32">
      <path d="M12 0C7.58 0 4 3.58 4 8c0 5.25 8 13 8 13s8-7.75 8-13c0-4.42-3.58-8-8-8zm0 11c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/>
    </svg>
  `),
  iconSize: [32, 32],
  iconAnchor: [16, 32],
});

const selectedIcon = new Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ef4444" width="40" height="40">
      <path d="M12 0C7.58 0 4 3.58 4 8c0 5.25 8 13 8 13s8-7.75 8-13c0-4.42-3.58-8-8-8zm0 11c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/>
    </svg>
  `),
  iconSize: [40, 40],
  iconAnchor: [20, 40],
});

// Component to handle map center changes
function MapController({ selectedCity, cities }: { selectedCity: string | null; cities: CityData[] }) {
  const map = useMap();

  useEffect(() => {
    // Disable zoom
    map.scrollWheelZoom.disable();
    map.doubleClickZoom.disable();
    map.touchZoom.disable();
    map.boxZoom.disable();
    map.keyboard.disable();

    // Disable dragging
    map.dragging.disable();
  }, [map]);

  return null;
}

export default function Map({ cities, onCitySelect, selectedCity, propertyType }: MapProps) {
  // Croatia center - optimized to show all cities in a square format
  const croatiaCenter: [number, number] = [44.5, 16.45];
  const fixedZoom = 8;

  const handleCityClick = (cityName: string) => {
    onCitySelect(cityName);

    // Auto-scroll to EDA section
    setTimeout(() => {
      const edaSection = document.getElementById('analiza');
      if (edaSection) {
        edaSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  // Filter out Solin
  const filteredCities = cities.filter(city => city.name !== 'Solin');

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
    <MapContainer
      center={croatiaCenter}
      zoom={fixedZoom}
      style={{ height: '1100px', width: '100%', background: '#0f172a', borderRadius: '12px' }}
      zoomControl={false}
      attributionControl={false}
    >
      {/* Dark tile layer */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
        opacity={0.4}
      />

      <MapController selectedCity={selectedCity} cities={filteredCities} />

      {filteredCities.map((city) => {
        const count = propertyType === 'apartments' ? city.apt_count : city.house_count;
        const isSelected = city.name === selectedCity;

        if (count === 0) return null;

        return (
          <Marker
            key={city.name}
            position={[city.lat, city.lng]}
            icon={isSelected ? selectedIcon : defaultIcon}
            eventHandlers={{
              click: () => handleCityClick(city.name),
            }}
          />
        );
      })}
    </MapContainer>
    </div>
  );
}
