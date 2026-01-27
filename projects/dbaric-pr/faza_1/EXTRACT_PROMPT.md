# GitHub PR Data Extraction Prompt

## Objective

Extract data from GitHub to predict PR workflow duration (time from open to merge/close).

**Note**: For draft PRs, workflow duration should be calculated from when the PR was marked as "ready_for_review" (not from `created_at`), as draft PRs are not actively in the review workflow.

## Required Data Points

### 1. PR Core Information

- **Endpoint**: `GET /repos/{owner}/{repo}/pulls/{pull_number}`
- **Fields to extract**:
  - `created_at` (PR open timestamp)
  - `merged_at` (merge timestamp, null if closed)
  - `closed_at` (close timestamp, null if open)
  - `title`
  - `body` (description)
  - `user.login` (author username)
  - `state` (open/closed/merged)
  - `number` (PR number)
  - `draft` (boolean - true if PR is in draft state)

### 1a. PR Timeline Events (for Draft Status)

- **Endpoint**: `GET /repos/{owner}/{repo}/issues/{issue_number}/timeline`
- **Note**: PRs are issues in GitHub, so use the PR number as `issue_number`
- **Fields to extract**:
  - Event type (filter for draft-related events like "ready_for_review" or "converted_to_draft")
  - Timestamp when draft status changed (`created_at` field in event)
- **Important**: Verify the exact event type names and response structure in the API response, as field names may vary
- **Alternative**: If timeline endpoint is unavailable or incomplete, use `created_at` from PR core info and check if `draft: false` to infer ready state

### 2. PR Revisions (Commits)

- **Endpoint**: `GET /repos/{owner}/{repo}/pulls/{pull_number}/commits`
- **Pagination**: Yes (use `per_page=100` and follow `Link` header for next page)
- **Fields per commit**:
  - `commit.author.date` (commit timestamp)
  - `commit.message` (commit message)
  - `author.login` (commit author username, may be null)
  - `commit.committer.date` (committer timestamp)

### 3. PR Issue Comments

- **Endpoint**: `GET /repos/{owner}/{repo}/issues/{issue_number}/comments`
- **Note**: PRs are issues, so use the PR number as `issue_number`
- **Pagination**: Yes (use `per_page=100` and follow `Link` header for next page)
- **Fields per comment**:
  - `created_at` (comment timestamp)
  - `user.login` (commenter username)
  - `body` (comment content)

### 4. PR Review Comments (Inline Code Comments)

- **Endpoint**: `GET /repos/{owner}/{repo}/pulls/{pull_number}/comments`
- **Pagination**: Yes (use `per_page=100` and follow `Link` header for next page)
- **Fields per review comment**:
  - `created_at` (comment timestamp)
  - `user.login` (reviewer username)
  - `body` (comment content)
  - `path` (file path where comment was made)
  - `line` (line number where comment was made)

### 5. PR Reviews

- **Endpoint**: `GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews`
- **Pagination**: Yes (use `per_page=100` and follow `Link` header for next page)
- **Fields per review**:
  - `submitted_at` (review timestamp)
  - `user.login` (reviewer username)
  - `state` (APPROVED/CHANGES_REQUESTED/COMMENTED/DISMISSED)

### 6. PR Requested Reviewers

- **Endpoint**: `GET /repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers`
- **Fields**:
  - `users[].login` (reviewer username)
  - `teams[].name` (team names if teams are requested)

## Implementation Notes

### Node.js Implementation

- Use ES modules (`import`/`export`) with top-level await
- Use Node.js built-in `fetch` (Node 18+) - no external libraries required
- Import from `fs/promises`: `import { readFile, writeFile } from "fs/promises"`
- Base URL: `https://api.github.com`
- Read token from `.env` file:
  ```javascript
  const token = (await readFile(".env", "utf8"))
    .match(/GITHUB_TOKEN=(.+)/)[1]
    .trim();
  ```
- Fetch with headers:
  ```javascript
  const res = await fetch(url, {
    headers: {
      Authorization: `token ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });
  ```
- Check status before parsing: `console.log("Status:", res.status)`
- Save results to JSON file: `await writeFile("filename.json", JSON.stringify(data, null, 2))`

### Pagination Handling

- Check response headers for `Link` header: `res.headers.get("Link")`
- Parse `Link` header to find `rel="next"` URL
- Continue fetching until no `next` link exists
- Example Link header format: `<https://api.github.com/...?page=2>; rel="next", <https://api.github.com/...?page=3>; rel="last"`
- Implementation pattern:
  ```javascript
  let allData = [];
  let url = initialUrl;
  while (url) {
    const res = await fetch(url, { headers });
    const data = await res.json();
    allData.push(...data);
    const linkHeader = res.headers.get("Link");
    url = linkHeader?.match(/<([^>]+)>; rel="next"/)?.[1];
  }
  ```

### Error Handling

- Check `res.status` before parsing JSON
- Handle rate limiting (429 status) with appropriate delays
- Handle 404 for missing PRs/issues

### Draft PR Handling

- Check `draft` field from PR core info
- If `draft: true`, fetch timeline events to find `ready_for_review` event
- Use `ready_for_review` event timestamp as workflow start time
- If `draft: false` or no timeline available, use `created_at` as workflow start time

## Output Format

Save all extracted data to a single JSON file using `writeFile`:

```javascript
const output = {
  pr: {
    /* extracted PR core info */
  },
  commits: [
    /* all commits */
  ],
  issue_comments: [
    /* all issue comments */
  ],
  review_comments: [
    /* all review comments */
  ],
  reviews: [
    /* all reviews */
  ],
  requested_reviewers: [
    /* requested reviewers */
  ],
  timeline_events: [
    /* draft status events */
  ],
};

await writeFile("pr-data.json", JSON.stringify(output, null, 2));
```

Structure:

- `pr`: Core PR information with extracted fields
- `commits`: Array of all commits (handle pagination)
- `issue_comments`: Array of all issue comments (handle pagination)
- `review_comments`: Array of all review comments (handle pagination)
- `reviews`: Array of all reviews (handle pagination)
- `requested_reviewers`: Array of requested reviewers
- `timeline_events`: Array of timeline events (for draft status tracking)

All timestamps should be preserved in ISO 8601 format. Include all usernames, timestamps, and content for analysis.
