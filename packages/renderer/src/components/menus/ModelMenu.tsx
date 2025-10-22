import React, { useEffect } from 'react';
import { useManifestTypes, useManifests, useManifestsByModel, useManifestsByType, useManifestsByModelAndType, useManifest } from '@/lib/manifest';

const ModelMenu = () => {
    const { data: manifests } = useManifests();

    useEffect(() => {
        console.log(manifests);
    }, [manifests]);
    
  return (
    <div>

    </div>
  );
};

export default ModelMenu;