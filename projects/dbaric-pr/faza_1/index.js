import { readFile, writeFile } from "fs/promises";

const OWNER = "[user]";
const REPO = "[project]";
const START_PR = 500;

const token = (await readFile(".env", "utf8"))
  .match(/GITHUB_TOKEN=(.+)/)[1]
  .trim();

const headers = {
  Authorization: `token ${token}`,
  "X-GitHub-Api-Version": "2022-11-28",
};

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchPaginated(url) {
  let allData = [];
  let currentUrl = url;
  while (currentUrl) {
    const res = await fetch(currentUrl, { headers });
    const data = await res.json();
    allData.push(...data);
    const linkHeader = res.headers.get("Link");
    currentUrl = linkHeader?.match(/<([^>]+)>; rel="next"/)?.[1];
    if (currentUrl) await delay(50);
  }
  return allData;
}

async function fetchPRData(prNumber, pr) {
  await delay(50);
  const reviewsUrl = `https://api.github.com/repos/${OWNER}/${REPO}/pulls/${prNumber}/reviews?per_page=100`;
  const reviews = await fetchPaginated(reviewsUrl);

  await delay(50);
  const requestedReviewersUrl = `https://api.github.com/repos/${OWNER}/${REPO}/pulls/${prNumber}/requested_reviewers`;
  const requestedReviewersRes = await fetch(requestedReviewersUrl, { headers });
  const requestedReviewers = await requestedReviewersRes.json();

  await delay(50);
  const timelineUrl = `https://api.github.com/repos/${OWNER}/${REPO}/issues/${prNumber}/timeline?per_page=100`;
  const timelineEvents = await fetchPaginated(timelineUrl);

  return {
    pr,
    reviews,
    requested_reviewers: requestedReviewers,
    timeline_events: timelineEvents,
  };
}

let allPRs = [];
let currentPR = START_PR;
let latestPR = null;

// Find latest PR number
const latestPRUrl = `https://api.github.com/repos/${OWNER}/${REPO}/pulls?state=all&per_page=1&sort=created&direction=desc`;
const latestPRRes = await fetch(latestPRUrl, { headers });
const latestPRData = await latestPRRes.json();
if (latestPRData.length > 0) {
  latestPR = latestPRData[0].number;
}

while (true) {
  await writeFile("state.txt", currentPR.toString());

  const prUrl = `https://api.github.com/repos/${OWNER}/${REPO}/pulls/${currentPR}`;
  const prRes = await fetch(prUrl, { headers });
  const status = prRes.status;

  console.log(`PR ${currentPR}: ${status}`);

  if (status === 404) {
    break;
  }

  const pr = await prRes.json();
  const prData = await fetchPRData(currentPR, pr);
  allPRs.push(prData);
  await writeFile("prs.json", JSON.stringify(allPRs, null, 2));

  if (latestPR && currentPR >= latestPR) {
    break;
  }

  currentPR++;
  await delay(50);
}

console.log(`Completed. Processed ${allPRs.length} PRs.`);
