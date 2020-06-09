"""
Show info about the redis job queue
"""
import argparse
from redis import Redis


# flake8: noqa: T001
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-list", action="store_true")
    args = parser.parse_args()

    r = Redis(decode_responses=True)

    def default_dict(job):
        job_set = set(map(int, r.lrange(job, 0, -1)))
        nr_jobs_total = len(job_set)
        return {"started": 0, "finished": 0, "jobs": job_set,
                "nr_jobs_total": nr_jobs_total}

    results = {}
    for key in r.keys():
        if ":" in key:
            job, nr_str = key.split(":")
            nr = int(nr_str)
            status = r.hget(key, "status")
            if job not in results:
                results[job] = default_dict(job)
            results[job][status] += 1
            results[job]["jobs"].remove(nr)
        else:
            if key not in results:
                results[key] = default_dict(key)

    FMT = "{:<20}{:<10}{:<10}{:<10}{:<10}"
    print(FMT.format("Job", "started", "finished", "total", "submitted"))
    print("-" * 60)
    for job, states in sorted(results.items(), key=lambda x: x[0]):
        print(FMT.format(job, states["started"], states["finished"],
                         states["started"] + states["finished"],
                         states["nr_jobs_total"]))

    if args.show_list:
        print("\n\nNeither started nor finished jobs")
        for job, state in results.items():
            print("\nJob", job)
            print(sorted(state["jobs"]))


if __name__ == "__main__":
    main()
