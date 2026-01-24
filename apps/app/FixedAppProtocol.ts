import fs from "node:fs";
// get the user data path

const userDataPath = "/Users/tosinkuye/Library/Application Support/Electron";

// open json 
const appRequests = fs.readFileSync("app-requests.json", "utf8").split("\n").filter(line => line.length > 0).map(line => JSON.parse(line));

// get unique urls
const uniqueUrls = new Set(appRequests.map(request => request.url));

for (const url of uniqueUrls) {
    // decode the url
    const urlObj = new URL(url);
    const path = decodeURIComponent(urlObj.pathname);
    if (fs.existsSync(path)) {
        console.log("Yay we found the file", path);
    } else {
        console.log("No file found", path);
    }
}