"""This source is passed to an isolated container, never imported with model code.

It is trusted harness source. Candidate and benchmark code execute ONLY inside
the container. One invocation processes one candidate and one MBPP+ task.
"""
import contextlib
import io
import json
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU,(8,8))
    request=json.load(sys.stdin)
    source=request['source'];public=request['public_tests'];plus=request['plus_test']
    # The candidate may print, including during imports. Contain ordinary
    # stdout/stderr; the caller separately caps raw Docker pipe output.
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            namespace={'__name__':'__rvl_candidate__'}
            exec(compile(source,'candidate.py','exec'),namespace)
            public_results=[]
            for test in public:
                try:exec(compile(test,'public_test.py','exec'),namespace)
                except BaseException:public_results.append(0)
                else:public_results.append(1)
        except BaseException:
            public_results=[0]*len(public)
        try:
            namespace={'__name__':'__rvl_candidate__'}
            exec(compile(source,'candidate.py','exec'),namespace)
            exec(compile(plus,'evaluation_only_test.py','exec'),namespace)
            trusted=1
        except BaseException:
            trusted=0
    print(json.dumps({'public_passes':public_results,'trusted_pass':trusted}),flush=True)


if __name__=='__main__':main()
